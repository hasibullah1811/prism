import { useState } from 'react'
import { Split, FileText, AlertTriangle, Search, Info, Download, Map } from 'lucide-react'
// NEW IMPORT
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

function App() {
  const [text, setText] = useState("")
  const [query, setQuery] = useState("") 
  const [chunks, setChunks] = useState([])
  const [queryCoords, setQueryCoords] = useState({x:0, y:0}) // NEW STATE
  const [loading, setLoading] = useState(false)
  
  const [chunkSize, setChunkSize] = useState(500)
  const [overlap, setOverlap] = useState(50)

  const handleProcess = async () => {
    setLoading(true)
    try {
      const response = await fetch("http://127.0.0.1:8000/process-text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          text: text || "RAG (Retrieval-Augmented Generation) is...", 
          query: query,
          chunk_size: chunkSize, 
          overlap: overlap 
        })
      })
      const data = await response.json()
      setChunks(data.chunks)
      setQueryCoords(data.query_coords) // Save Query Location
    } catch (error) {
      console.error("Error:", error)
      alert("Is Python running?")
    }
    setLoading(false)
  }

  const handleExportJSON = () => {
    if (chunks.length === 0) return;
    const jsonString = JSON.stringify(chunks, null, 2);
    const blob = new Blob([jsonString], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "prism-export.json";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans p-8">
      
      {/* HEADER */}
      <header className="max-w-6xl mx-auto mb-8 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="bg-white p-2 rounded-lg border border-slate-200 shadow-sm">
            <Split className="w-6 h-6 text-blue-600" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight">Prism</h1>
            <p className="text-xs text-slate-500 font-medium">RAG VISUALIZER</p>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* LEFT COLUMN: Controls */}
        <div className="lg:col-span-4 space-y-6">
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
            <h2 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-4">Configuration</h2>
            
            <div className="mb-6">
               <label className="text-sm font-medium text-slate-700 mb-1 block">Search Simulation</label>
               <div className="relative">
                 <Search className="absolute left-3 top-2.5 w-4 h-4 text-slate-400" />
                 <input 
                   type="text" 
                   placeholder="Enter a query to test retrieval..." 
                   className="w-full pl-9 pr-3 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/20"
                   value={query}
                   onChange={(e) => setQuery(e.target.value)}
                 />
               </div>
            </div>
            
            <hr className="border-slate-100 my-4"/>

            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="font-medium text-slate-700">Chunk Size</span>
                  <span className="text-slate-500">{chunkSize}</span>
                </div>
                <input type="range" min="100" max="2000" step="50" value={chunkSize} onChange={(e) => setChunkSize(Number(e.target.value))} className="w-full h-2 bg-slate-100 rounded-lg appearance-none cursor-pointer accent-blue-600" />
              </div>
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="font-medium text-slate-700">Overlap</span>
                  <span className="text-slate-500">{overlap}</span>
                </div>
                <input type="range" min="0" max="500" step="10" value={overlap} onChange={(e) => setOverlap(Number(e.target.value))} className="w-full h-2 bg-slate-100 rounded-lg appearance-none cursor-pointer accent-blue-600" />
              </div>
            </div>
          </div>

          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm flex flex-col h-[400px]">
            <h2 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-4 flex items-center gap-2">
              <FileText className="w-3 h-3" /> Source Text
            </h2>
            <textarea 
              className="flex-1 w-full bg-slate-50 border border-slate-200 rounded-lg p-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/20 resize-none"
              placeholder="Paste your document here..."
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
            <button onClick={handleProcess} disabled={loading} className="mt-4 w-full bg-slate-900 hover:bg-slate-800 text-white font-medium py-2.5 rounded-lg transition-all">
              {loading ? "Processing..." : "Visualize"}
            </button>
          </div>
        </div>

        {/* RIGHT COLUMN: Results */}
        <div className="lg:col-span-8 space-y-6">
           
           {/* --- 1. SEMANTIC MAP (New Visualization) --- */}
           {chunks.length > 2 && (
             <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
                <h2 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-4 flex items-center gap-2">
                  <Map className="w-3 h-3" /> Semantic Map (PCA)
                </h2>
                <div className="h-64 w-full bg-slate-50 rounded-lg border border-slate-100 relative">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" dataKey="x" name="x" hide />
                      <YAxis type="number" dataKey="y" name="y" hide />
                      <Tooltip cursor={{ strokeDasharray: '3 3' }} content={({ payload }) => {
                          if (payload && payload.length) {
                            const data = payload[0].payload;
                            return (
                              <div className="bg-white p-2 border shadow-sm rounded text-xs">
                                <p className="font-bold">{data.label}</p>
                                {data.score && <p>Match: {Math.round(data.score * 100)}%</p>}
                              </div>
                            );
                          }
                          return null;
                      }} />
                      {/* CHUNKS (Blue Dots) */}
                      <Scatter name="Chunks" data={chunks} fill="#3b82f6">
                         {chunks.map((entry, index) => (
                           <Cell key={`cell-${index}`} fill={entry.score > 0.05 ? '#10b981' : '#cbd5e1'} />
                         ))}
                      </Scatter>
                      {/* QUERY (Red Dot) */}
                      {query && (
                        <Scatter name="Query" data={[{ x: queryCoords.x, y: queryCoords.y, label: "QUERY" }]} fill="#ef4444" shape="star" />
                      )}
                    </ScatterChart>
                  </ResponsiveContainer>
                  <p className="absolute bottom-2 right-2 text-[10px] text-slate-400 bg-white/50 px-2 rounded">
                    Closer dots = More similar meaning
                  </p>
                </div>
             </div>
           )}

           {/* --- 2. LIST VIEW --- */}
           {chunks.length === 0 ? (
             <div className="h-full flex flex-col items-center justify-center text-slate-400 border-2 border-dashed border-slate-200 rounded-xl min-h-[400px]">
                <Info className="w-8 h-8 mb-2 opacity-50" />
                <p>Enter text and click Process</p>
             </div>
           ) : (
             <div className="space-y-4">
               <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <h2 className="text-lg font-semibold text-slate-800">Results</h2>
                    <span className="text-xs bg-slate-200 text-slate-600 px-2 py-0.5 rounded-full">
                      {chunks.length}
                    </span>
                  </div>
                  <button onClick={handleExportJSON} className="flex items-center gap-2 text-sm font-medium text-slate-600 hover:text-blue-600 bg-white border border-slate-200 hover:border-blue-200 px-3 py-1.5 rounded-lg shadow-sm transition-all">
                    <Download className="w-4 h-4" />
                    Export JSON
                  </button>
               </div>
               
               {chunks.map((chunk) => {
                 const isMatch = chunk.score > 0.05; 
                 return (
                 <div key={chunk.id} className={`group bg-white rounded-xl border shadow-sm transition-all duration-200 overflow-hidden ${isMatch ? 'border-emerald-500 ring-4 ring-emerald-500/5' : 'border-slate-200 hover:shadow-md'}`}>
                    <div className={`px-5 py-3 border-b flex items-center justify-between ${isMatch ? 'bg-emerald-50 border-emerald-100' : 'bg-slate-50/50 border-slate-50'}`}>
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">Chunk {chunk.id}</span>
                      </div>
                      <div className="flex items-center gap-2">
                         {isMatch && (
                           <span className="text-[10px] font-bold bg-emerald-100 text-emerald-700 px-2 py-1 rounded-full border border-emerald-200">
                             MATCH {Math.round(chunk.score * 100)}%
                           </span>
                         )}
                         {chunk.bad_cut && (
                           <span className="flex items-center gap-1 text-[10px] font-bold bg-amber-50 text-amber-600 px-2 py-1 rounded-full border border-amber-100">
                             <AlertTriangle className="w-3 h-3" /> BAD CUT
                           </span>
                         )}
                         <span className="text-[10px] font-bold bg-slate-100 text-slate-500 px-2 py-1 rounded-full border border-slate-200">
                           {chunk.tokens} TOKENS
                         </span>
                      </div>
                    </div>
                    <div className="p-5 text-sm leading-relaxed text-slate-700 font-medium">
                      {chunk.overlap && (
                        <span className="bg-yellow-100 text-yellow-800 border-b-2 border-yellow-300 rounded-sm px-1 mx-0.5" title="Overlap">
                          {chunk.overlap}
                        </span>
                      )}
                      {chunk.remaining}
                    </div>
                 </div>
               )})}
             </div>
           )}
        </div>
      </main>
    </div>
  )
}

export default App