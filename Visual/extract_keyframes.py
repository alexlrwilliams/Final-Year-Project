from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter
import os

if __name__ == "__main__":
  vd = Video()
  max_frames_return = 12

  root_dir_path = "../data/videos"
  context_dir_path = os.path.join(root_dir_path, "context")
  utterance_dir_path = os.path.join(root_dir_path,"utterance")
  frames_dir_path = os.path.join("..","data","frames", "utterance")

  diskwriter = KeyFrameDiskWriter(location=frames_dir_path)

  vd.extract_keyframes_from_videos_dir(
       no_of_frames=max_frames_return, dir_path=context_dir_path,
       writer=diskwriter
  )

  vd.extract_keyframes_from_videos_dir(
      no_of_frames=max_frames_return, dir_path=utterance_dir_path,
      writer=diskwriter
  )


