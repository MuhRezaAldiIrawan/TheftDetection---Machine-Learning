import os
from pickle import FALSE, TRUE
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
sys.path.insert(0, './yolov5')
import argparse
import os
import platform
import shutil
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
# from yolov5.models.experimental import attempt_load
# from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, 
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
count1 = 0
data = []
def gen(opt):
    """Video streaming generator function."""

    out, source1, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, project, name, exist_ok = \
        opt.output, opt.source1, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.name, opt.exist_ok
    webcam = source1 == '0' or source1.startswith(
        'rtsp') or source1.startswith('http') or source1.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(deep_sort_model,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Directories
    save_dir = increment_path(Path(project) / name,
                              exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    # half precision only supported by PyTorch on CUDA
    half &= pt and device.type != 'cpu'
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source1, img_size=imgsz,
                              stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source1, img_size=imgsz,
                             stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # extract what is in between the last '/' and last '.'
    txt_file_name = source1.split('/')[-1].split('.')[0]
    txt_path = str(Path(save_dir)) + '/' + txt_file_name + '.txt'

    if pt and device.type != 'cpu':
        model(torch.zeros(
            1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(
            save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
        pred = model(img, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
        dt[2] += time_sync() - t3
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            s += '%gx%g ' % img.shape[2:]  # print string

            annotator = Annotator(im0, line_width=2, pil=not ascii)
            w, h = im0.shape[1], im0.shape[0]
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # add to string
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                t4 = time_sync()
                outputs = deepsort.update(
                    xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)):

                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        # count
                        # bbox_left = output[0]
                        # bbox_top = output[1]
                        # bbox_w = output[2] - output[0]
                        # bbox_h = output[3] - output[1]
                        c = int(cls)  # integer class
                        label = f'{id}'
                        if names[c] == 'car' or names[c] == 'truck' or names[c] == 'bus':
                            dir2 = save_dir / 'count' / f'{id}.jpg'
                            count_obj1(bboxes, w, h, id, dir2, im0, save_txt, txt_path)
                            
                        annotator.box_label(bboxes, label, color=colors(c, True))
                        #     count_car(bboxes, w, h, id)
                        # if names[c] == 'truck':
                        #     count_truck(bboxes, w, h, id)
                        # if names[c] == 'bus':
                        #     count_truck(bboxes, w, h, id)
                        # if names[c] == 'person' or names[c] == 'motorcycle' or names[c] == 'bicycle':
                        #     dir2 = save_dir / 'person' / f'{id}.jpg'
                        #     gambar = f'{id}.jpg'
                        #     l = str(Path(save_dir))
                        #     simpan=l.lstrip('static')
                        #     # simpan = dir3 / gambar
                        #     count_pelanggaran(bboxes, w, h, id, dir2, im0,gambar, simpan)
                        

            #     LOGGER.info(
            #         f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')

            # else:
            #     deepsort.increment_ages()
            #     LOGGER.info('No detections')

            # Stream results
            im0 = annotator.result()
            if show_vid:
                global count1
                color = (0, 255, 0)
                start_point = (429,123)
                end_point = (412,227)
                start_point1 = (494,124)
                end_point1 = (430,123)
                start_point2 = (415,227)
                end_point2 = (496,232)
                start_point3 = (493,127)
                end_point3 = (495,231)
                cv2.line(im0,(266,350), (426,89), color, 3)
                cv2.line(im0,(556,353), (546,237), color, 3)
                # cv2.rectangle(im0,(429,123), (412,227), color, 3)
                cv2.line(im0, start_point, end_point, color, thickness=2)
                cv2.line(im0, start_point1, end_point1, color, thickness=2)
                cv2.line(im0, start_point2, end_point2, color, thickness=2)
                cv2.line(im0, start_point3, end_point3, color, thickness=2)
                thickness = 2
                org = (20, 30)
                # org1 = (20, 60)
                # org2 = (20, 90)
                # org3 = (20, 115)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                cv2.putText(im0, 'Total = ' + str(count1), org, font,
                            fontScale, color, thickness, cv2.LINE_AA)
                
                # print('total ='+str(count1))
                # cv2.putText(im0, 'Mobil = ' + str(car1), org1, font,
                #             fontScale, color, thickness, cv2.LINE_AA)
                # cv2.putText(im0, 'Truk dan bus = ' + str(truck1), org2, font,
                #             fontScale, color, thickness, cv2.LINE_AA)
                # cv2.putText(im0, 'Pelanggaran = ' + str(person), org3, font,
                #             fontScale, color, thickness, cv2.LINE_AA)
                # cv2.putText(im0,'bus= ' + str(bus), org3, font,
                # fontScale, color, thickness, cv2.LINE_AA)
                cv2.imshow(str(p), im0)
                # frame = cv2.imencode('.jpg', im0)[1].tobytes()
                # yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            # if save_txt:
                            
            #     # # to MOT format
            
            #     # Write MOT compliant results to file
            #     with open(txt_path, 'a') as f:
            #         # f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
            #         #                                bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))
            #         f.write('Total = '+str(count1))
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 15, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'
                    vid_writer = cv2.VideoWriter(
                        save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
        per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        print('Results saved to %s' % save_path)
        if platform == 'win':  # MacOS
            os.system('open ' + save_path)


def count_obj1(box, w, h, id, dir2, im0, save_txt, txt_path):
    global count1, data
    center_coordinates = (
        int(box[0]+(box[2]-box[0])/2), int(box[1]+(box[3]-box[1])/2))
    if int(box[1]+(box[3]-box[1])/2) > (115) and int(box[1]+(box[3]-box[1])/2) < (220) and int(box[0]+(box[2]-box[0])/2) < (340) and int(box[0]+(box[2]-box[0])/2) > (230):
        cv2.putText(im0, 'Cordinate = ' + str(int(box[0]+(box[2]-box[0])/2))+','+str(int(box[1]+(box[3]-box[1])/2)), (20,80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (100, 255 ,230), 2, cv2.LINE_AA)
        cv2.line(im0, (int(box[0]),int(box[1])),(int(box[2]),int(box[3])) , (255,0,0), thickness=2)
        # cv2.line(im0, ((int(box[1]),int(box[0])),(int(box[3]),int(box[2]))) , (255,0,0), thickness=2)
        if id not in data:
            count1 += 1
            data.append(id)
            print('cordinate: '+str(int(box[0]+(box[2]-box[0])/2))+','+str(int(box[1]+(box[3]-box[1])/2)))
            if save_txt:
                with open(txt_path, 'w') as f:
                    print('Total = '+str(count1))
                    f.write('Total = '+str(count1))
                    # save_one_box(box, im0, file=dir2, BGR=True)
           
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str,
                        default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    # file/folder, 0 for webcam
    parser.add_argument('--source1', type=str,
                        default='videos/dataset2.mp4', help='source')
    parser.add_argument('--output', type=str, default='inference/output',
                        help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+',
                        type=int, default=[480], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float,
                        default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='cpu',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_false',
                        help='display tracking video results')
    parser.add_argument('--save_vid', action='store_true',
                        help='save video tracking results')
    parser.add_argument('--save_txt', action='store_false',
                        help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 1 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--evaluate', action='store_true',
                        help='augmented inference')
    parser.add_argument("--config_deepsort", type=str,
                        default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true",
                        help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true',
                        help='visualize features')
    parser.add_argument('--max-det', type=int, default=2000,
                        help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true',
                        help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT /
                        'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='cctv_gerbang',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    with torch.no_grad():
        gen(opt)