import cv2
import os

def capture_document_from_camera(save_filename="camera_photo.jpg"):
    # カメラを起動します（0番はパソコンに最初から付いている標準のカメラです）
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("【エラー】カメラが見つからないみたいです…接続などを確認してみてくださいね。")
        return None

    print("=========================================")
    print(" カメラが起動しました！")
    print(" 画面に書類を映して、「スペースキー」を押すと撮影します。")
    print(" やめたい時は半角の「q」キーを押してくださいね。")
    print("=========================================")

    saved_path = None

    while True:
        # カメラから映像を1コマずつ読み込みます
        ret, frame = cap.read()
        if not ret:
            print("映像が読み込めませんでした…")
            break

        # 画面に映像を表示します
        cv2.imshow("Scanner (Press Space to capture, Q to quit)", frame)

        # キーボードの入力を待ちます
        key = cv2.waitKey(1) & 0xFF

        # スペースキー（ASCIIコード32）が押されたら撮影です！
        if key == 32:
            # 今このプログラムがあるフォルダの場所を取得します
            base_dir = os.path.dirname(os.path.abspath(__file__))
            saved_path = os.path.join(base_dir, save_filename)
            
            # 画像ファイルとして保存します
            cv2.imwrite(saved_path, frame)
            print(f"\n🌟 パシャッ！ 写真を撮りました！")
            print(f"保存先: {saved_path}")
            break
        
        # 'q'キーが押されたら終了します
        elif key == ord('q'):
            print("\n撮影をキャンセルしました。")
            break

    # カメラを閉じて、画面も片付けます
    cap.release()
    cv2.destroyAllWindows()
    
    return saved_path

# ==========================================
#   ここから実行部分です
# ==========================================

if __name__ == "__main__":
    # 関数を呼び出して、撮影をスタートします
    captured_file = capture_document_from_camera()
    
    if captured_file and os.path.exists(captured_file):
        print("\n無事に撮影が終わりました！ 次はこの画像をNDLOCR-Liteに渡しますね。")
