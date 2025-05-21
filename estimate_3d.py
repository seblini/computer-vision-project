import cv2
import mediapipe as mp
import argparse
import numpy as np

mp_pose = mp.solutions.pose
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS

def compute_joint_angle(e1, joint, e3):
    A = e1 - joint
    B = e3 - joint

    cos_theta = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)

    return np.degrees(angle_rad)

def fuse_lms(norm1, norm2, world1, world2):
    if norm1.visibility + norm2.visibility == 0:
        return None

    x = (
        world1.x * norm1.visibility * 0.75 + world2.z * norm2.visibility * 0.25
    ) / (
        norm1.visibility * 0.75 + norm2.visibility * 0.25
    )

    y = (
        world1.y * norm1.visibility + world2.y * norm2.visibility
    ) / (
        norm1.visibility + norm2.visibility
    )

    z = (
        world1.z * norm1.visibility * 0.25 - world2.x * norm2.visibility * 0.75
    ) / (
        norm1.visibility * 0.25 + norm2.visibility * 0.75
    )

    return np.array([x, y, z])

def run_analysis(frame1, frame2, norms1, norms2, worlds1, worlds2):
    fused = np.array([fuse_lms(*zipped) for zipped in zip(norms1, norms2, worlds1, worlds2)])

    right_arm_angle = compute_joint_angle(fused[16], fused[14], fused[12])
    left_arm_angle = compute_joint_angle(fused[15], fused[13], fused[11])

    right_shoulder_angle = compute_joint_angle(fused[14], fused[12], fused[24])
    left_shoulder_angle = compute_joint_angle(fused[13], fused[11], fused[23])

    left_wrist_angle = compute_joint_angle(np.mean(fused[17:19:2,:], axis=0), fused[15], fused[13])
    right_wrist_angle = compute_joint_angle(np.mean(fused[18:20:2,:], axis=0), fused[16], fused[14])

    arm_sync = np.abs(right_arm_angle - left_arm_angle) < 20

    full_extension = ((right_shoulder_angle + left_shoulder_angle)/2) > 130 and \
    ((right_arm_angle + left_arm_angle)/2) > 150

    right_tucked_elbow = np.abs(fused[12, 0] - fused[14, 0]) < 0.07
    left_tucked_elbow = np.abs(fused[11, 0] - fused[13, 0]) < 0.07

    guide_hand_straight = left_wrist_angle > 150
    shooting_hand_down = np.abs(right_wrist_angle - 90) < 20

    cv2.putText(frame2, "Full Extension", (20, frame2.shape[0]-50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if arm_sync and full_extension else (0,0,255), 2)

    cv2.putText(frame2, f"Guide Hand Angle: {left_wrist_angle:.0f}", (20, frame2.shape[0]-80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if guide_hand_straight else (0,0,255), 2)

    cv2.putText(frame2, f"Shooting Hand Angle: {right_wrist_angle:.0f}", (20, frame2.shape[0]-110),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if shooting_hand_down else (0,0,255), 2)

    cv2.putText(frame2, "Tucked Elbows", (20, frame2.shape[0]-140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if right_tucked_elbow and left_tucked_elbow else (0,0,255), 2)

def draw_pose(frame, landmarks, visibility_thresh=0.5):
    h, w, _ = frame.shape

    for idx, lm in enumerate(landmarks):
        if idx < 11:
            continue

        if lm.visibility > visibility_thresh:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)

    for i1, i2 in POSE_CONNECTIONS:
        if i1 < 11 or i2 < 11:
            continue

        lm1 = landmarks[i1]
        lm2 = landmarks[i2]
        if lm1.visibility > visibility_thresh and lm2.visibility > visibility_thresh:
            x1, y1 = int(lm1.x * w), int(lm1.y * h)
            x2, y2 = int(lm2.x * w), int(lm2.y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)


def run_dual_pose(video1, video2, out1, out2):
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out_vid1 = cv2.VideoWriter(out1, fourcc, fps, (width, height))
    out_vid2 = cv2.VideoWriter(out2, fourcc, fps, (width, height))

    pose1 = mp_pose.Pose()
    pose2 = mp_pose.Pose()

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break

        rgb1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        res1 = pose1.process(rgb1)
        res2 = pose2.process(rgb2)

        if res1.pose_landmarks and \
            res2.pose_landmarks and \
            res1.pose_world_landmarks and \
            res2.pose_world_landmarks:

            draw_pose(frame1, res1.pose_landmarks.landmark)
            draw_pose(frame2, res2.pose_landmarks.landmark)

            run_analysis(
                frame1,
                frame2,
                res1.pose_landmarks.landmark,
                res2.pose_landmarks.landmark,
                res1.pose_world_landmarks.landmark,
                res2.pose_world_landmarks.landmark
            )

        out_vid1.write(frame1)
        out_vid2.write(frame2)

        combined = cv2.hconcat([frame1, frame2])
        cv2.imshow("Pose Estimation", combined)
        if cv2.waitKey(15) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    out_vid1.release()
    out_vid2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video1", required=True)
    parser.add_argument("--video2", required=True)
    parser.add_argument("--out1", default="estimate/pose_cam1_world.mp4")
    parser.add_argument("--out2", default="estimate/pose_cam2_world.mp4")
    args = parser.parse_args()

    run_dual_pose(args.video1, args.video2, args.out1, args.out2)
