import cv2
import numpy as np
import os
from tqdm import tqdm
import os.path as osp


def get_captures(video_name:list,path:str):
    video_capture = [cv2.VideoCapture(osp.join(path,name)) for name in video_name]
    for capture in video_capture:
        assert capture.isOpened()
    return video_capture


def set_frame_pos(video_capture:list,currentFrame):
    back_frame = max(currentFrame - 31, 0)  # Keyframes every 30 frames
    # print('back_frame set to: {0}'.format(back_frame))
    for i,capture in enumerate(video_capture):
        capture.set(cv2.CAP_PROP_POS_FRAMES, back_frame)
        if not capture.get(cv2.CAP_PROP_POS_FRAMES) == back_frame:
            print(
                'Warning: OpenCV has failed to set video {3} back_frame to {0}. OpenCV value is {1}. Target value is {2}'.format(
                    back_frame, capture.get(cv2.CAP_PROP_POS_FRAMES), currentFrame,i))

        back_frame = capture.get(cv2.CAP_PROP_POS_FRAMES)
        # print('back_frame is: {0}'.format(back_frame))
        while back_frame < currentFrame:
            capture.read()
            back_frame += 1


def video_reader(video_name, path, start_pos = 1,end_pos = 30000):
    video_capture = get_captures(video_name,path)
    set_frame_pos(video_capture, start_pos)
    while start_pos < end_pos:
        start_pos += 1
        yield [capture.read()[1] for capture in video_capture]


def warning_iter(warnings,video_num = 4, max_iter = 10000):
    duration = 100
    max_thickness = 30
    color_range = [150,250]
    color_c = (color_range[1]-color_range[0])/(duration/2)
    thickness_c = max_thickness/(duration/2)


    start = [duration]*video_num
    for i in range(max_iter):
        draw_warnings = [False]*video_num
        colors = [(0, 0, 0)]*video_num
        thickness = [1]*video_num
        if len(warnings)>0 and i == warnings[0][0]:
            videos_id = warnings[0][1]
            warnings.pop(0)
            for v_id in videos_id:
                # start[v_id] = 0 if start[v_id]>=60 else start[v_id]
                start[v_id] = 0

        for i in range(video_num):
            if start[i]<duration:
                draw_warnings[i]=True
                if start[i]<duration/2:
                    colors[i]=(0,0,color_range[0] + int(color_c*start[i]))
                    thickness[i] = 4 + int(thickness_c*start[i])
                else:
                    colors[i] = (0, 0, color_range[0] + int((color_range[1]-color_range[0])*2 - color_c * start[i]))
                    thickness[i] = 4 + int(max_thickness*2 -  thickness_c*start[i])
        start = [i+1 for i in start]
        yield draw_warnings,colors,thickness


if __name__ == '__main__':
    videos = ['fragment_original_0.avi',
              'fragment_presearch_0.avi',
              'fragment_weighted_0.avi',
              'fragment_weighted_presearch_0.avi']
    labels = [
        'Baseline',
        'K Times Forward Search',
        'Weight Average',
        'K Times Forward Search & Weight Average'
    ]
    coords = [
        (800, 100),
        (550, 100),
        (780, 100),
        (100, 100),
    ]
    warning_pos = [
        (256,[0,1,2]),
        (648,[0,2]),
        (896,[0,2]),
        (923,[0,1,2])
    ]


    path = '/data/stu06/homedir/project/gluon/MCMT/pipeline'
    border = 1
    sub_width = 1920//2 - border
    sub_height = 1080//2 - border
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter('compare.avi',
                                        fourcc, 59.94, (1920, 1080))
    video_iter = video_reader(videos,path,5100,6400)
    warning_i = warning_iter(warning_pos)
    frame_id = 0
    for frames,warnings in tqdm(zip(video_iter,warning_i)):
        # frame = frames[0]
        frames = [cv2.putText(frame, label, coord, cv2.FONT_HERSHEY_SIMPLEX, 2.5,
                              (0,255,255), thickness=4) for coord,label,frame in zip(coords,labels,frames)]


        for i,(draw_warning,color,thickness) in enumerate(zip(*warnings)):
            if draw_warning:
                frames[i] = cv2.rectangle(frames[i],(0,0),(1920,1080),color,thickness)

        frames = [cv2.resize(frame, (sub_width, sub_height)) for frame in frames]

        frame = np.zeros(shape=(1080,1920,3),dtype=np.int)
        for i in range(2):
            for j in range(2):
                idx = i*2+j
                img = frames[idx]
                frame[i*(sub_height+border):i*(sub_height+border)+sub_height,
                j * (sub_width + border):j * (sub_width + border) + sub_width,:]=img
        video_writer.write(frame.astype(np.uint8))
