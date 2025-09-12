# Transcribe + Split Pod

Inputs (event.input):
- job_id (required): unique pipeline id.
- local_video_path (optional): defaults to /storage/{job_id}/raw/full.mp4
- input_url (optional): if provided, the pod downloads the video to local_video_path.
- language (optional): default "en".
- split_phrase (optional): defaults from env SPLIT_PHRASE_DEFAULT or "sermon".
- announcements_phrase (optional): defaults from env ANNOUNCEMENTS_PHRASE_DEFAULT or "".

Outputs:
- transcripts: transcript.txt, transcript.json, timestamped.txt
- splits: worship.mp4 (if split found), sermon.mp4, announcements.mp4 (if found)
