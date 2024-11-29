import cv2
import turtle
import threading

# Turtle 화살표 생성
def turtle_arrow():
    screen = turtle.Screen()
    screen.setup(width=800, height=600)
    screen.title("Turtle 화살표")
    screen.bgcolor("white")

    arrow = turtle.Turtle()
    arrow.shape("triangle")
    arrow.shapesize(stretch_wid=1, stretch_len=3)
    arrow.penup()
    arrow.speed(0)

    # 초기 화살표 방향
    arrow.setheading(90)  # 위쪽
    arrow.color("green")
    arrow.goto(0, 0)
    arrow.stamp()

    # 이벤트 루프 실행
    turtle.done()

# OpenCV 비디오 캡처
def opencv_video():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 화면에 텍스트 추가
        cv2.putText(frame, "OpenCV Active", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 비디오 출력
        cv2.imshow('OpenCV Window', frame)

        # 'q' 키로 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 멀티스레딩 실행
if __name__ == "__main__":
    turtle_thread = threading.Thread(target=turtle_arrow)
    opencv_thread = threading.Thread(target=opencv_video)

    turtle_thread.start()
    opencv_thread.start()

    turtle_thread.join()
    opencv_thread.join()
