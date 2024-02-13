import cv2

video_path = "dvor.mp4"  # замените на реальный путь к вашему видео

cap = cv2.VideoCapture(video_path)
tracker = None  # Инициализация трекера
roi_coords = None  # Координаты ROI
prev_roi_center = None  # Переменная для хранения предыдущего центра ROI


def start_tracking(frame):
    global tracker, roi_coords, prev_roi_center
    # Set the ROI (Region of Interest)
    x, y, w, h = cv2.selectROI(frame)

    # Сохраняем предыдущие координаты ROI
    prev_roi_coords = roi_coords
    prev_roi_center = (x + w // 2, y + h // 2)

    # Initialize the tracker
    tracker = cv2.legacy.TrackerMOSSE_create()
    tracker.init(frame, (x, y, w, h))

    # Сохраняем новые координаты ROI
    roi_coords = (x, y, w, h)

    # Если у нас уже есть предыдущие координаты, добавляем смещение
    if prev_roi_coords is not None:
        x += prev_roi_coords[0]
        y += prev_roi_coords[1]

    cv2.destroyAllWindows()


def reset_tracking():
    global tracker, roi_coords
    tracker = None
    roi_coords = None


def stabilize_roi(frame, roi_center):
    global prev_roi_center, roi_coords
    if roi_coords is not None and prev_roi_center is not None:
        # Рассчитываем смещение между текущим и предыдущим центрами ROI
        dx = roi_center[0] - prev_roi_center[0]
        dy = roi_center[1] - prev_roi_center[1]

        # Применяем смещение к координатам ROI
        x, y, w, h = roi_coords
        x += dx
        y += dy

        # Обновляем координаты ROI
        roi_coords = (x, y, w, h)

        # Пересчитываем центр ROI после обновления координат
        roi_center = (x + w // 2, y + h // 2)

        # Отрисовываем контур на стабилизированном кадре
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        margin = 20
        # Получаем новый стабилизированный кадр с обновленными координатами ROI
        stabilized_frame = frame[max(y - margin, 0):min(y + h + margin, frame.shape[0]),
                           max(x - margin, 0):min(x + w + margin, frame.shape[1])].copy()

        return stabilized_frame

    return None


def main():
    global prev_roi_center
    key = 0  # Переменная для хранения кода последней нажатой клавиши
    percent = 78
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        width = int(frame.shape[1] * percent / 100)
        height = int(frame.shape[0] * percent / 100)
        dim = (width, height)
        frame = cv2.resize(frame, dim)

        # Если трекера нет, отображаем текущий кадр
        if tracker is None:
            cv2.imshow('Video', frame)

        # Если трекера нет и нажата клавиша 's', начинаем выделение
        if key == ord('s') and tracker is None:
            start_tracking(frame)

        # Если нажата клавиша 'r', сбрасываем трекер
        elif key == ord('r'):
            reset_tracking()
            cv2.destroyAllWindows()

        elif tracker is not None:
            ret, track_window = tracker.update(frame)
            x, y, w, h = int(track_window[0]), int(track_window[1]), int(track_window[2]), int(track_window[3])
            roi_center = (x + w // 2, y + h // 2)  # Центр ROI

            stabilized_frame = stabilize_roi(frame, roi_center)
            # Отображаем стабилизированный кадр
            if stabilized_frame is not None and stabilized_frame.shape[0] > 0 and stabilized_frame.shape[1] > 0:
                cv2.imshow('Stabilized Video', stabilized_frame)
                cv2.imshow('Video', frame)

            prev_roi_center = roi_center

        # Выход из трекинга при нажатии клавиши 'q'
        elif key == ord('q'):
            break

        key = cv2.waitKey(20)  # Ожидание клавиш и сохранение кода последней клавиши

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
