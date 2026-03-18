import cv2
import os
import subprocess
import glob

def capture_document_from_camera(save_filename="camera_photo.jpg"):
    """カメラで書類を撮影して保存する関数です"""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("【エラー】カメラが見つからないみたいです…")
        return None

    print("=========================================")
    print(" 📷 カメラが起動しました！")
    print(" 画面に書類を映して、「スペースキー」で撮影します。")
    print("=========================================")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    saved_path = os.path.join(base_dir, save_filename)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Scanner (Press Space to capture)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 32: # スペースキー
            cv2.imwrite(saved_path, frame)
            print(f"\n🌟 パシャッ！ 写真を撮りました！")
            break
        elif key == ord('q'):
            print("\n撮影をキャンセルしました。")
            saved_path = None
            break

    cap.release()
    cv2.destroyAllWindows()
    return saved_path

def run_ndlocr(image_path, ndlocr_exe_path):
    """NDLOCR-Liteを呼び出して文字を読み取る関数です"""
    print("\n📝 NDLOCR-Liteにお願いして、文字を読んでもらっています…")
    print("（少しお時間がかかるかもしれません。がんばれー…！）")
    
    # テキストを保存するフォルダを作ります
    base_dir = os.path.dirname(image_path)
    output_dir = os.path.join(base_dir, "ocr_result")
    os.makedirs(output_dir, exist_ok=True)

    # NDLOCR-Liteをコマンドラインから動かすための命令文を作ります
    # ※ NDLOCR-Liteの仕様に合わせて引数（-i や -o など）は調整が必要です
    # 一般的なコマンドラインツールの書き方にしています
    command = f'"{ndlocr_exe_path}" -i "{image_path}" -o "{output_dir}"'

    try:
        # ここで別のソフト（NDLOCR）を裏で実行します！
        subprocess.run(command, shell=True, check=True)
        
        # 出来上がったテキストファイルを探します
        txt_files = glob.glob(os.path.join(output_dir, "*.txt"))
        if not txt_files:
            print("【エラー】テキストファイルが作られなかったみたいです…")
            return None
            
        # 中身の文字を読み込みます
        with open(txt_files[0], 'r', encoding='utf-8') as f:
            text_result = f.read()
            
        return text_result

    except Exception as e:
        print(f"【エラー】NDLOCR-Liteの実行中に問題が起きてしまいました…: {e}")
        return None

# ==========================================
#   ここから実行部分です
# ==========================================

if __name__ == "__main__":
    # ★重要：篤志さんのパソコンにある、NDLOCR-Liteの実行ファイル(.exe や .bat)の
    # フルパスに書き換えてくださいね！
    ndlocr_path = r"C:\path\to\ndlocr_lite\ndlocr_cli.exe" 
    
    # 1. カメラで撮影します
    captured_file = capture_document_from_camera()
    
    if captured_file:
        # 2. 撮った写真をNDLOCR-Liteに渡します
        extracted_text = run_ndlocr(captured_file, ndlocr_path)
        
        if extracted_text:
            print("\n=========================================")
            print(" ✨ 読み取り完了です！こんな文字が見つかりました ✨")
            print("-----------------------------------------")
            print(extracted_text)
            print("=========================================")
            
            # 3. キーワードがあるかチェックします
            if "契約書" in extracted_text or "請求書" in extracted_text:
                print("\n💮 「契約書」や「請求書」という文字が入っています！ 大成功です！")
            else:
                print("\n💦 キーワードは見つかりませんでした…")
