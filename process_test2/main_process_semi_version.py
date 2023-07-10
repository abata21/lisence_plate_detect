import cv2
from multiprocessing import Process, Queue, active_children, Pipe
import main_process_load_video as mplv
from concurrent.futures import ThreadPoolExecutor
import lp_detection_tracking
import numpy as np
import time


#
class Multi_processing_rtsp_cls():
    def __init__(self):
        """
        각 동용상에 대한 Queue, Pipe, video_path init
        각각의 Queue와 Pipe는 sub.py에 input_rtsp 함수에 연결되어 있음

        -> 이전에는 실행됬는데 여기다가 yolov7 init이 안됨
           현재 sub.py에서 각각의 프로세스마다 yolov7을 init하여 사용
           -> 수정중에 있음

          """


        self.frame_q1 = Queue()
        self.detect_q1 = Queue()
        self.fps_q1 = Queue()
        self.video1 = r'C:\Users\hsson.DI-SOLUTION\Desktop\video1\green_1cam_2.mp4'

        self.frame_q2 = Queue()
        self.detect_q2 = Queue()
        self.fps_q2 = Queue()
        self.video2 = r'C:\Users\hsson.DI-SOLUTION\Desktop\video2\white_1cam_1.mp4'

        self.frame_q3 = Queue()
        self.detect_q3 = Queue()
        self.fps_q3 = Queue()
        self.video3 = r'C:\Users\hsson.DI-SOLUTION\Desktop\video2\yellow_2_line1.mp4'


        self.object_stop_parent_pipe1, self.object_stop_child_pipe1 = Pipe()
        self.object_stop_parent_pipe2, self.object_stop_child_pipe2 = Pipe()
        self.object_stop_parent_pipe3, self.object_stop_child_pipe3 = Pipe()


    def multi_processing_rtsp(self):

        total_list= []  # processs가 pipe에서 end 통신을 받기전에 종료될 경우 process 재생성시 이름 부여하기 위해 (lien 73~74)


        Processes1 = Process(target=mplv.input_rtsp,
                             args=(self.video1, self.frame_q1, self.detect_q1, self.fps_q1, self.object_stop_child_pipe1),
                             name='Processes1')
        Processes2 = Process(target=mplv.input_rtsp,
                             args=(self.video2, self.frame_q2, self.detect_q2, self.fps_q2, self.object_stop_child_pipe2),
                             name='Processes2')
        Processes3 = Process(target=mplv.input_rtsp,
                             args=(self.video3, self.frame_q3, self.detect_q3, self.fps_q3, self.object_stop_child_pipe3),
                             name='Processes3')

        executor = ThreadPoolExecutor(2)
        executor.submit(self.video_show1)
        executor = ThreadPoolExecutor(2)
        executor.submit(self.video_show2)
        executor = ThreadPoolExecutor(2)
        executor.submit(self.video_show3)

        Processes1.daemon = True
        Processes1.start()
        Processes2.daemon = True
        Processes2.start()
        Processes3.daemon = True
        Processes3.start()

        for act_chil in active_children():
            total_list.append(act_chil)

        """
            다시 살아나는지 확인하려고 죽여 본 것
            -> 다시 살아나는데 시간소요가 좀 있음
                 """

        time.sleep(15)

        Processes2.terminate()
        Processes2.kill()

        time.sleep(1)

        Processes1.terminate()
        Processes1.kill()

        # while True:
        #     fps1 = int(self.fps_q1.get())
        #     fps2 = int(self.fps_q2.get())
        #     fps3 = int(self.fps_q3.get())
        #
        #     print('fps1',fps1)
        #     print('fps2', fps2)
        #     print('fps3', fps3)



        while True:
            """
            강제 종료인지 일반 종료인지 알기 위해 Pipe 송신 여부 확인 후 강제 종료일 경우 그 프로세스와 같은 인자의 새로운 프로세스 생성 후 실행
            일반 종료일 경우 실행된 순서대로 종료
              """
            # 강제 종료 경우
            if not self.object_stop_parent_pipe1.poll() and not self.object_stop_parent_pipe2.poll() and not self.object_stop_parent_pipe3.poll():
                if not Processes1.is_alive() and not self.object_stop_parent_pipe1.poll():
                    new_name = 'Processes' + str(len(total_list) + 1)  #Process의 이름이 겹치면 안되서 새로운 이름 생성
                    total_list.append(new_name)
                    Processes1 = Process(target=mplv.input_rtsp,
                                         args=(self.video1, self.frame_q1, self.detect_q1, self.fps_q1,
                                               self.object_stop_child_pipe1),
                                         name=new_name)
                    Processes1.daemon = True
                    Processes1.start()

                elif not Processes2.is_alive() and not self.object_stop_parent_pipe2.poll():
                    new_name2 = 'Processes' + str(len(total_list) + 1)
                    total_list.append(new_name2)
                    Processes2 = Process(target=mplv.input_rtsp,
                                         args=(self.video2, self.frame_q2, self.detect_q2, self.fps_q2,
                                               self.object_stop_child_pipe2),
                                         name=new_name2)
                    Processes2.daemon = True
                    Processes2.start()

                elif not Processes3.is_alive() and not self.object_stop_parent_pipe3.poll():
                    new_name3 = 'Processes' + str(len(total_list) + 1)
                    total_list.append(new_name3)
                    Processes3 = Process(target=mplv.input_rtsp,
                                         args=(self.video3, self.frame_q3, self.detect_q3, self.fps_q3,
                                               self.object_stop_child_pipe3),
                                         name=new_name3)
                    Processes3.daemon = True
                    Processes3.start()


            # 일반 종료 경우
            else:
                if str(self.object_stop_parent_pipe1.poll()):
                    msg = self.object_stop_parent_pipe1.recv()
                    print(active_children())
                    if msg == 'end':
                        print('video1_get_meg')
                        self.frame_q1 = None
                        self.detect_q1 = None
                        self.fps_q1 = None
                        self.video_show1 = False
                        time.sleep(1)
                        Processes1.terminate()
                        print('first', Processes1.is_alive())
                        if Processes1.is_alive():
                            Processes1.kill()
                            print(Processes1.is_alive())
                else:
                    pass

                if str(self.object_stop_parent_pipe2.poll()):
                    msg = self.object_stop_parent_pipe2.recv()
                    if msg == 'end':
                        print('video2_get_meg')
                        self.frame_q2 = None
                        self.detect_q2 = None
                        self.fps_q2 = None
                        self.video_show2 = False
                        time.sleep(1)
                        Processes2.terminate()
                        print('second', Processes2.is_alive())
                        if Processes2.is_alive():
                            Processes2.kill()
                            print(Processes2.is_alive())
                else:
                    pass

                if str(self.object_stop_parent_pipe3.poll()):
                    msg = self.object_stop_parent_pipe3.recv()
                    if msg == 'end':
                        print('video3_get_meg')
                        self.frame_q3 = None
                        self.detect_q3 = None
                        self.fps_q3 = None
                        self.video_show3 = False
                        time.sleep(1)
                        Processes3.terminate()
                        print('thired', Processes3.is_alive())
                        if Processes3.is_alive():
                            Processes3.kill()
                            print(Processes3.is_alive())
                else:
                    pass

            if len(active_children()) == 0:
                print('동영상_끝')
                break

    def video_show1(self):
        while True:
            if self.frame_q1.qsize() > 0 and self.detect_q1.qsize() > 0:
                original_img = self.frame_q1.get()
                infer_img = self.detect_q1.get()
                resize_original_img = cv2.resize(original_img, (640, 480))
                resize_infer_img = cv2.resize(infer_img, (640, 480))
                fps_str = self.fps_q1.get()

                # Check FPS
                cv2.putText(resize_original_img, fps_str, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

                # Convert to one Window
                numpy_horizontal_img = np.hstack((resize_original_img, resize_infer_img))
                window_name = str('video1')
                cv2.imshow(window_name, numpy_horizontal_img)
                cv2.waitKey(1)

    def video_show2(self):
        while True:
            if self.frame_q2.qsize() > 0 and self.detect_q2.qsize() > 0:
                original_img = self.frame_q2.get()
                infer_img = self.detect_q2.get()
                resize_original_img = cv2.resize(original_img, (640, 480))
                resize_infer_img = cv2.resize(infer_img, (640, 480))
                fps_str = self.fps_q2.get()

                # Check FPS
                cv2.putText(resize_original_img, fps_str, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

                # Convert to one Window
                numpy_horizontal_img = np.hstack((resize_original_img, resize_infer_img))
                window_name = str('video2')
                cv2.imshow(window_name, numpy_horizontal_img)
                cv2.waitKey(1)

    def video_show3(self):
        while True:
            if self.frame_q3.qsize() > 0 and self.detect_q3.qsize() > 0:
                original_img = self.frame_q3.get()
                infer_img = self.detect_q3.get()
                resize_original_img = cv2.resize(original_img, (640, 480))
                resize_infer_img = cv2.resize(infer_img, (640, 480))
                fps_str = self.fps_q3.get()

                # Check FPS
                cv2.putText(resize_original_img, fps_str, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

                # Convert to one Window
                numpy_horizontal_img = np.hstack((resize_original_img, resize_infer_img))
                window_name = str('video3')
                cv2.imshow(window_name, numpy_horizontal_img)
                cv2.waitKey(1)




def main():
    multi_rtsp = Multi_processing_rtsp_cls()
    multi_rtsp.multi_processing_rtsp()

if __name__ == '__main__':
    main()
