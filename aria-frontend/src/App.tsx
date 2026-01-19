import React, { useRef, useState, useEffect } from 'react';
import { Upload, Loader2, Image as ImageIcon, Languages, Clipboard, GripVertical, RefreshCw, X } from 'lucide-react';

interface TranslationResult {
  text: string;
  translatedText: string;
  box: [number, number, number, number];
}

export default function App() {
  const [loading, setLoading] = useState(false);
  const [retranslating, setRetranslating] = useState(false);
  const [results, setResults] = useState<TranslationResult[]>([]);
  const [imagePreview, setImagePreview] = useState<string>("");
  const [currentFile, setCurrentFile] = useState<File | null>(null);
  const [draggedIndex, setDraggedIndex] = useState<number | null>(null);
  const [headerVisible, setHeaderVisible] = useState(true);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const lastScrollY = useRef(0);
  
  // Your live Cloudflare Worker URL
  const WORKER_URL = "https://aria-backend.lijack466.workers.dev";

  // Render canvas when image and results are available
  useEffect(() => {
    if (imagePreview && currentFile && results.length > 0 && canvasRef.current) {
      renderManga(currentFile, results);
    }
  }, [imagePreview, currentFile, results]);

  // Handle paste from clipboard
  useEffect(() => {
    const handlePaste = async (e: ClipboardEvent) => {
      const items = e.clipboardData?.items;
      if (!items) return;

      for (let i = 0; i < items.length; i++) {
        if (items[i].type.indexOf('image') !== -1) {
          e.preventDefault();
          const blob = items[i].getAsFile();
          if (blob) {
            // Create a File object from the blob
            const file = new File([blob], 'pasted-image.png', { type: blob.type });
            await processImage(file);
          }
          break;
        }
      }
    };

    window.addEventListener('paste', handlePaste);
    return () => window.removeEventListener('paste', handlePaste);
  }, []);

  // Auto-hide header on scroll down
  useEffect(() => {
    const handleScroll = () => {
      const currentScrollY = window.scrollY;
      
      if (currentScrollY < 10) {
        // Always show at top of page
        setHeaderVisible(true);
      } else if (currentScrollY > lastScrollY.current) {
        // Scrolling down - hide header
        setHeaderVisible(false);
      } else {
        // Scrolling up - show header
        setHeaderVisible(true);
      }
      
      lastScrollY.current = currentScrollY;
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const processImage = async (file: File) => {
    setLoading(true);
    const formData = new FormData();
    formData.append("image", file);

    try {
      const resp = await fetch(WORKER_URL, { method: "POST", body: formData });
      
      if (!resp.ok) {
        const errorData = await resp.json().catch(() => ({ error: "Unknown error" }));
        console.error("Worker error:", errorData);
        const errorDetails = errorData.details ? `\n\nDetails: ${errorData.details}` : '';
        const errorStack = errorData.stack ? `\n\nStack: ${errorData.stack}` : '';
        throw new Error(`Worker error: ${errorData.error || resp.statusText}${errorDetails}${errorStack}`);
      }
      
      const data = await resp.json();
      console.log("Worker response:", data);
      
      if (!data.results || data.results.length === 0) {
        const debugInfo = data.debug ? `\n\nDebug info:\n${data.debug.message}\n\nRaw AI response: ${data.debug.rawResponse}` : '';
        alert(`No Japanese text detected in the image. Try another image with clearer text.${debugInfo}`);
        console.log("Full debug data:", data);
        return;
      }
      
      setCurrentFile(file);
      setResults(data.results);
      setImagePreview(URL.createObjectURL(file));
    } catch (err: any) {
      console.error("Full error:", err);
      alert(`Error: ${err.message || "Unknown error occurred"}`);
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    await processImage(file);
  };

  const handleDragStart = (index: number) => {
    setDraggedIndex(index);
  };

  const handleDragOver = (e: React.DragEvent, index: number) => {
    e.preventDefault();
    if (draggedIndex === null || draggedIndex === index) return;
    
    // Reorder array
    const newResults = [...results];
    const draggedItem = newResults[draggedIndex];
    newResults.splice(draggedIndex, 1);
    newResults.splice(index, 0, draggedItem);
    
    setResults(newResults);
    setDraggedIndex(index);
  };

  const handleDragEnd = () => {
    setDraggedIndex(null);
    // Re-render canvas with new order
    if (currentFile && results.length > 0) {
      renderManga(currentFile, results);
    }
  };

  const handleDelete = (index: number) => {
    const newResults = results.filter((_, i) => i !== index);
    setResults(newResults);
    // Re-render canvas with remaining bubbles
    if (currentFile && newResults.length > 0) {
      renderManga(currentFile, newResults);
    }
  };

  const retranslate = async () => {
    if (results.length === 0) return;
    
    setRetranslating(true);
    try {
      // Extract Japanese text in current order
      const textsToTranslate = results.map(r => r.text);
      
      const resp = await fetch(`${WORKER_URL}/retranslate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ texts: textsToTranslate })
      });

      if (!resp.ok) {
        throw new Error('Retranslation failed');
      }

      const data = await resp.json();
      
      // Update translations while preserving order and boxes
      const updatedResults = results.map((result, index) => ({
        ...result,
        translatedText: data.translations[index]
      }));
      
      setResults(updatedResults);
    } catch (err: any) {
      console.error('Retranslation error:', err);
      alert(`Retranslation failed: ${err.message}`);
    } finally {
      setRetranslating(false);
    }
  };

  const renderManga = (file: File, results: TranslationResult[]) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      // Set canvas dimensions to match image
      canvas.width = img.width;
      canvas.height = img.height;
      
      // Clear canvas and draw image
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);

      results.forEach((res, index) => {
        const [ymin, xmin, ymax, xmax] = res.box;
        const x = xmin;
        const y = ymin;
        const width = xmax - xmin;
        const height = ymax - ymin;

        // Draw numbered box outline (no fill - keep original image visible)
        ctx.strokeStyle = "#f43f5e";
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, width, height);
        
        // Draw number badge
        ctx.fillStyle = "#f43f5e";
        ctx.fillRect(x, y - 30, 35, 30);
        ctx.fillStyle = "white";
        ctx.font = "bold 18px sans-serif";
        ctx.fillText((index + 1).toString(), x + 10, y - 8);
      });
    };
    img.onerror = () => {
      console.error("Failed to load image");
    };
    img.src = URL.createObjectURL(file);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* Header */}
      <header className={`border-b border-slate-800/50 backdrop-blur-sm bg-slate-950/50 sticky top-0 z-10 transition-transform duration-300 ${
        headerVisible ? 'translate-y-0' : '-translate-y-full'
      }`}>
        <div className="container mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="bg-gradient-to-br from-rose-500 to-pink-600 p-3 rounded-xl shadow-lg shadow-rose-500/20">
                <Languages className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-rose-400 to-pink-400 bg-clip-text text-transparent">
                  ARIA
                </h1>
                <p className="text-sm text-slate-400">Manga Translation Tool</p>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-6 py-8">
        {/* Upload Section */}
        {!imagePreview && (
          <div className="max-w-2xl mx-auto">
            <div 
              onClick={() => fileInputRef.current?.click()}
              className="relative border-2 border-dashed border-slate-700 hover:border-rose-500/50 rounded-2xl p-12 text-center cursor-pointer transition-all duration-300 hover:bg-slate-900/50 group"
            >
              <input 
                ref={fileInputRef}
                type="file" 
                accept="image/*" 
                onChange={handleUpload}
                className="hidden"
              />
              
              <div className="flex flex-col items-center gap-4">
                <div className="p-6 rounded-full bg-slate-800 group-hover:bg-rose-500/10 transition-colors">
                  {loading ? (
                    <Loader2 className="w-12 h-12 text-rose-400 animate-spin" />
                  ) : (
                    <Upload className="w-12 h-12 text-slate-400 group-hover:text-rose-400 transition-colors" />
                  )}
                </div>
                
                <div>
                  <h3 className="text-xl font-semibold mb-2">
                    {loading ? "Analyzing your manga..." : "Upload a manga page"}
                  </h3>
                  <p className="text-slate-400 text-sm">
                    {loading 
                      ? "Detecting and translating Japanese text..." 
                      : "Click to select, drag and drop, or paste an image (Ctrl+V)"
                    }
                  </p>
                </div>
                
                {!loading && (
                  <div className="flex items-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-rose-500 to-pink-600 text-white font-medium shadow-lg shadow-rose-500/20">
                    <ImageIcon className="w-4 h-4" />
                    Choose File
                  </div>
                )}
              </div>
            </div>
            
            <div className="mt-8 p-6 rounded-xl bg-slate-900/50 border border-slate-800">
              <h4 className="font-semibold mb-3 flex items-center gap-2">
                <Languages className="w-5 h-5 text-rose-400" />
                How it works
              </h4>
              <ul className="space-y-2 text-sm text-slate-400">
                <li className="flex items-start gap-2">
                  <span className="text-rose-400 mt-0.5">•</span>
                  <span>Upload any manga page with Japanese text</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-rose-400 mt-0.5">•</span>
                  <span>AI automatically detects text regions</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-rose-400 mt-0.5">•</span>
                  <span>Get instant translations with visual overlays</span>
                </li>
              </ul>
              
              <div className="mt-4 pt-4 border-t border-slate-700">
                <div className="flex items-center gap-2 text-sm text-slate-300">
                  <Clipboard className="w-4 h-4 text-emerald-400" />
                  <span className="font-medium">Pro tip:</span>
                  <span className="text-slate-400">Press Ctrl+V (or Cmd+V) to paste from clipboard</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Results Section */}
        {imagePreview && (
          <div className="flex flex-col lg:flex-row gap-6">
            {/* Image Panel */}
            <div className="flex-1">
              <div className="bg-slate-900/50 rounded-2xl p-6 border border-slate-800">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold">Analyzed Image</h3>
                  <button 
                    onClick={() => {
                      setImagePreview("");
                      setResults([]);
                      setCurrentFile(null);
                      if (fileInputRef.current) fileInputRef.current.value = "";
                    }}
                    className="px-3 py-1.5 text-sm rounded-lg bg-slate-800 hover:bg-slate-700 transition-colors"
                  >
                    Upload New
                  </button>
                </div>
                <div className="relative bg-slate-800 rounded-lg overflow-hidden min-h-[400px] flex items-center justify-center">
                  <canvas 
                    ref={canvasRef} 
                    className="max-w-full h-auto block" 
                    style={{ imageRendering: 'auto' }}
                  />
                </div>
              </div>
            </div>

            {/* Translations Panel */}
            {results.length > 0 && (
              <div className="lg:w-[420px]">
                <div className={`bg-slate-900/50 rounded-2xl p-6 border border-slate-800 sticky transition-all duration-300 ${
                  headerVisible ? 'top-24' : 'top-4'
                }`}>
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-lg font-semibold flex items-center gap-2">
                      <Languages className="w-5 h-5 text-rose-400" />
                      Translations ({results.length})
                    </h3>
                    <button
                      onClick={retranslate}
                      disabled={retranslating}
                      className="flex items-center gap-1 px-3 py-1.5 text-xs rounded-lg bg-emerald-600 hover:bg-emerald-500 transition-colors text-white font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                      title="Retranslate with current order for better context"
                    >
                      {retranslating ? (
                        <Loader2 className="w-3 h-3 animate-spin" />
                      ) : (
                        <RefreshCw className="w-3 h-3" />
                      )}
                      {retranslating ? 'Retranslating...' : 'Retranslate'}
                    </button>
                  </div>
                  
                  <div className="space-y-4 max-h-[calc(100vh-240px)] overflow-y-auto scrollbar-thin pr-2">
                    {results.map((res, index) => (
                      <div 
                        key={index}
                        draggable
                        onDragStart={() => handleDragStart(index)}
                        onDragOver={(e) => handleDragOver(e, index)}
                        onDragEnd={handleDragEnd}
                        className={`bg-slate-800/50 rounded-xl p-4 border border-slate-700/50 hover:border-rose-500/30 transition-all cursor-move ${
                          draggedIndex === index ? 'opacity-50 scale-95' : ''
                        }`}
                      >
                        <div className="flex items-center justify-between mb-3">
                          <div className="flex items-center gap-3">
                            <GripVertical className="w-4 h-4 text-slate-600 flex-shrink-0" />
                            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-rose-500 to-pink-600 flex items-center justify-center text-sm font-bold shadow-lg shadow-rose-500/20">
                              {index + 1}
                            </div>
                            <span className="text-xs text-slate-400 uppercase tracking-wider">Original</span>
                          </div>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDelete(index);
                            }}
                            className="p-1 rounded-lg hover:bg-red-500/20 text-slate-400 hover:text-red-400 transition-colors"
                            title="Delete this text box"
                          >
                            <X className="w-4 h-4" />
                          </button>
                        </div>
                        
                        <p className="text-base mb-4 text-slate-200 font-medium leading-relaxed">
                          {res.text}
                        </p>
                        
                        <div className="pt-3 border-t border-slate-700">
                          <span className="text-xs text-slate-400 uppercase tracking-wider">Translation</span>
                          <p className="text-base mt-2 text-emerald-400 leading-relaxed">
                            {res.translatedText}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
