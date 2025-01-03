import serial
import time

# 시리얼 포트와 속도 설정 (포트 이름은 환경에 맞게 변경)
arduino = serial.Serial(port='COM6', baudrate=9600, timeout=.1)
time.sleep(2)  # 초기화 대기

def send_command(command):
    print(command)
    print(type(command))
    arduino.write(f"{command}\n".encode())  # 명령 전송
    time.sleep(0.01)  # 약간의 대기

    # 응답 받기
    response = arduino.readline().decode('utf-8').strip()  # 응답 읽기

    if response:
        print(f"arduino: {response}")
    return response


