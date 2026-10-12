import React, { useState, useRef } from "react";

function VideoDetection() {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [processedVideoUrl, setProcessedVideoUrl] = useState("");
  const [loading, setLoading] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setVideoFile(file);
      setProcessedVideoUrl("");
    }
  };

  const handleUpload = async () => {
    if (!videoFile) return;
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("video", videoFile);

      // Gọi API đến Backend
      const response = await fetch("http://localhost:8000/analyze-video", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("AI Server error!");

      const data = await response.json();

      // Chỉ cập nhật URL video để hiển thị
      setProcessedVideoUrl(`http://localhost:8000${data.video_url}`);

    } catch (error) {
      console.error("Lỗi kết nối Backend:", error);
      alert("Không thể kết nối với Server AI!");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-100 via-white to-blue-50 flex items-center justify-center p-6 font-sans">
      <div className="w-full max-w-6xl bg-white rounded-3xl shadow-2xl border border-blue-100 p-8">

        <div className="text-center mb-10">
          <h1 className="text-4xl font-extrabold text-blue-700 tracking-tight">
            TRAFFIC AI SYSTEM
          </h1>
          <p className="text-gray-500 mt-2 font-medium">
            Object Detection & Lane Segmentation (ADAS Logic)
          </p>
        </div>

        {/* Upload Section */}
        <div className="group relative border-2 border-dashed border-blue-300 rounded-2xl p-10 text-center hover:bg-blue-50/50 transition-all cursor-pointer">
          <input 
            type="file" 
            accept="video/*" 
            onChange={handleFileChange} 
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          />
          <div className="space-y-2">
            <p className="text-blue-600 font-bold text-lg">
              {videoFile ? `${videoFile.name}` : "Upload Traffic Video"}
            </p>
            <p className="text-gray-400 text-sm">Support: MP4</p>
          </div>
        </div>

        {/* Start Button */}
        <div className="text-center mt-8">
          <button
            onClick={handleUpload}
            disabled={!videoFile || loading}
            className={`px-12 py-4 rounded-2xl shadow-xl font-black text-white tracking-widest transition-all active:scale-95 ${
              loading ? "bg-gray-400" : "bg-blue-600 hover:bg-blue-700"
            }`}
          >
            {loading ? "AI PROCESSING..." : "START ADAS ANALYSIS"}
          </button>
        </div>

        {/* Video Display */}
        <div className="relative mt-12 flex justify-center group">
          {processedVideoUrl && (
            <div className="relative rounded-3xl overflow-hidden shadow-2xl border-8 border-slate-100">
              <video 
                ref={videoRef} 
                src={processedVideoUrl} 
                autoPlay 
                loop 
                muted 
                controls 
                className="w-[900px]" 
              />
            </div>
          )}
        </div>

        {/* Metadata Footer */}
        <div className="mt-12 flex justify-between items-center px-4 border-t border-gray-100 pt-6">
          <div className="flex gap-4">
            <span className="text-[10px] bg-blue-100 text-blue-700 px-4 py-1.5 rounded-full font-bold uppercase tracking-widest">
              YOLOv8 Detection
            </span>
            <span className="text-[10px] bg-indigo-100 text-indigo-700 px-4 py-1.5 rounded-full font-bold uppercase tracking-widest">
              ResNet18-Unet Lane
            </span>
          </div>
          <p className="text-gray-300 text-[10px] font-mono tracking-tighter uppercase">
            ADAS Pipeline v1.0.6 - Minimal Build
          </p>
        </div>
      </div>
    </div>
  );
}

export default VideoDetection;