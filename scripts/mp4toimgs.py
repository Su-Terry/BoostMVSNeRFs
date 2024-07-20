if __name__ == '__main__':
    import sys
    import os
    import shutil
    import cv2

    if len(sys.argv) < 2:
        print('Usage: python scripts/mp4toimgs.py <your_video.mp4>')
        sys.exit(1)

    video = sys.argv[1]
    video_name = os.path.basename(video).split('.')[0]
    workspace = os.environ['workspace']
    out_dir = os.path.join(workspace, 'custom', 'images')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        shutil.rmtree(out_dir)
        os.makedirs(out_dir)

    cap = cv2.VideoCapture(video)
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(out_dir, f'{video_name}_{i:06d}.jpg'), frame)
        i += 1
    cap.release()
    print(f'Extracted {i} frames from {video}')
    print(f'Frames are saved in {out_dir}')