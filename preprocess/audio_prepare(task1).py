import os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import logging
import json
from scipy import signal

# 設定日誌記錄
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AudioProcessor:
    """音訊處理類 - 僅處理音訊，保持輸入輸出格式一致"""
    
    def __init__(self):
        """初始化音訊處理器"""
        # 優化後的配置
        self.config = {
            'normalize': True,
            'reduce_silence': True,
            'silence_threshold': -60,
            'silence_reduce_factor': 0.1,
            'denoise': True,
            'denoise_strength': 0.5,
            'n_fft': 1024,
            'hop_length': 256,
            'sr': 16000
        }
        
    def process_audio(self, input_file, output_file):
        """處理單個音訊檔案"""
        try:
            # 載入音訊
            audio, sr = librosa.load(input_file, sr=self.config['sr'])
            logger.info(f"載入音訊: {input_file}, 樣本數: {len(audio)}, 取樣率: {sr}")
            
            # 歸一化
            if self.config['normalize']:
                audio = librosa.util.normalize(audio)
                logger.info("已應用歸一化")
            
            # 降低靜音音量
            if self.config['reduce_silence']:
                audio = self._reduce_silence_volume(audio)
                logger.info("已降低靜音音量")
            
            # 降噪
            if self.config['denoise']:
                audio = self._denoise_audio(audio)
                logger.info("已應用降噪")
            
            # 保存音訊
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            sf.write(output_file, audio, sr)
            logger.info(f"音訊保存至: {output_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"處理音訊檔案失敗: {e}")
            return False
    
    def _reduce_silence_volume(self, audio):
        """降低靜音部分音量"""
        # 計算每個樣本的分貝值
        db = librosa.amplitude_to_db(np.abs(audio), ref=np.max)
        
        # 創建靜音掩碼
        silence_mask = db < self.config['silence_threshold']
        
        # 創建處理後的音訊數據副本
        processed_audio = audio.copy()
        
        # 降低靜音部分的音量
        processed_audio[silence_mask] *= self.config['silence_reduce_factor']
        
        logger.info(f"降低靜音部分音量: 檢測到 {np.sum(silence_mask)} 個靜音樣本 (共 {len(audio)} 個)")
        
        return processed_audio
    
    def _denoise_audio(self, audio):
        """
        使用頻譜減法進行音訊降噪，動態調整降噪強度
        """
        # 短時傅立葉變換參數
        n_fft = self.config['n_fft']
        hop_length = self.config['hop_length']
        
        # 計算短時傅立葉變換
        D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        
        # 計算功率譜
        power_spectrum = np.abs(D) ** 2
        
        # 估計噪音特徵
        # 假設前些幀為靜音（噪音），使用更少的幀避免誤將有效訊號視為噪音
        noise_frames = 5  # 從10減少到5
        noise_estimate = np.mean(power_spectrum[:, :noise_frames], axis=1, keepdims=True)
        
        # 計算信噪比
        snr = power_spectrum / (noise_estimate + 1e-10)
        
        # 計算全域SNR用於動態調整降噪強度
        global_snr_db = 10 * np.log10(np.mean(power_spectrum) / (np.mean(noise_estimate) + 1e-10))
        
        # 根據全域SNR動態調整降噪強度
        # 噪音越大（SNR越小），降噪強度越大
        base_strength = self.config['denoise_strength']
        if global_snr_db > 20:  # 信噪比高，背景很乾淨
            dynamic_strength = base_strength * 0.5  # 從0.7降為0.5，進一步減弱降噪
            logger.info(f"檢測到高信噪比 {global_snr_db:.2f}dB，減弱降噪強度")
        elif global_snr_db > 10:  # 信噪比中等
            dynamic_strength = base_strength * 0.8  # 從1.0降為0.8，適度減弱降噪
            logger.info(f"檢測到中等信噪比 {global_snr_db:.2f}dB，使用標準降噪強度")
        else:  # 信噪比低，噪音大
            dynamic_strength = base_strength * 1.2  # 從1.5降為1.2，降低過度降噪
            logger.info(f"檢測到低信噪比 {global_snr_db:.2f}dB，增強降噪強度")
        
        # 頻率相關的降噪強度調整
        # 低頻通常需要較弱的降噪，高頻通常需要較強的降噪
        # 調整頻率權重，減少中頻（語音頻段）的降噪強度
        freq_strength = np.zeros(D.shape[0])
        mid_freq_idx = int(D.shape[0] * 0.3)  # 語音頻段起始位置
        high_freq_idx = int(D.shape[0] * 0.7)  # 語音頻段結束位置
        
        # 保留語音頻段，減少降噪對語音的影響
        freq_strength[:mid_freq_idx] = np.linspace(0.6, 0.4, mid_freq_idx)  # 低頻降噪
        freq_strength[mid_freq_idx:high_freq_idx] = 0.3  # 語音頻段輕微降噪
        freq_strength[high_freq_idx:] = np.linspace(0.6, 1.0, D.shape[0] - high_freq_idx)  # 高頻降噪
        
        freq_strength = freq_strength.reshape(-1, 1) * dynamic_strength  # 應用動態強度
        
        # 應用頻譜減法
        gain = (snr - 1) / snr
        gain = np.maximum(gain, 0)  # 非負增益
        
        # 平滑處理以減少音樂噪聲
        gain = signal.medfilt2d(gain, kernel_size=3)
        
        # 應用不同頻率的降噪增益，避免過度降噪
        for i in range(gain.shape[0]):
            # 增加下限閾值，確保不會過度消除信號
            gain[i, :] = gain[i, :] ** freq_strength[i]
            gain[i, :] = np.maximum(gain[i, :], 0.1)  # 確保每個位置的增益至少為0.1
        
        # 應用增益到原始頻譜
        D_denoised = D * gain
        
        # 逆短時傅立葉變換
        audio_denoised = librosa.istft(D_denoised, hop_length=hop_length, length=len(audio))
        
        logger.info(f"噪音消除完成，基礎降噪強度: {base_strength}，動態調整後強度: {dynamic_strength:.2f}")
        
        return audio_denoised
    
    def process_dataset(self, input_dir, output_dir):
        """處理整個資料集"""
        # 創建輸出目錄
        os.makedirs(output_dir, exist_ok=True)
        
        # 獲取所有音訊檔案
        audio_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
        
        logger.info(f"開始處理 {len(audio_files)} 個音訊檔案")
        
        success_count = 0
        
        for audio_file in tqdm(audio_files, desc="處理音訊檔案"):
            input_path = os.path.join(input_dir, audio_file)
            output_path = os.path.join(output_dir, audio_file)
            
            if self.process_audio(input_path, output_path):
                success_count += 1
        
        logger.info(f"成功處理 {success_count}/{len(audio_files)} 個檔案")
        return success_count


def main():
    """主函數"""
    try:
        # 設置目錄路徑
        train1_audio_dir = os.path.join('.', 'dataset', 'TRAINGING_DATASET_1', 'Training_Dataset_01', 'audio')
        train1_output = os.path.join('.', 'prepared_audio', 'train1')
        
        train2_audio_dir = os.path.join('.', 'dataset', 'Training_Dataset_02', 'audio')
        train2_output = os.path.join('.', 'prepared_audio', 'train2')
        
        validation_audio_dir = os.path.join('.', 'dataset', 'TRAINGING_DATASET_1', 'Validation_Dataset', 'audio')
        validation_output = os.path.join('.', 'prepared_audio', 'validation')
        
        # 創建輸出目錄
        os.makedirs(train1_output, exist_ok=True)
        os.makedirs(train2_output, exist_ok=True)
        os.makedirs(validation_output, exist_ok=True)
        
        # 創建音頻處理器
        processor = AudioProcessor()
        
        # 處理訓練集1
        print("開始處理訓練集1...")
        processor.process_dataset(train1_audio_dir, train1_output)
        
        # 處理訓練集2
        print("開始處理訓練集2...")
        processor.process_dataset(train2_audio_dir, train2_output)
        
        # 處理驗證集
        print("開始處理驗證集...")
        processor.process_dataset(validation_audio_dir, validation_output)
        
        print("音頻處理完成")
        
    except Exception as e:
        print(f"處理失敗: {e}")
        logger.error(f"處理失敗: {e}")


if __name__ == "__main__":
    print("開始執行音訊處理...")
    main() 