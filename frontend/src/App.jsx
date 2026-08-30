import { useState } from 'react'

function App() {
  const [query, setQuery] = useState('')
  const [chatLog, setChatLog] = useState([])
  const [loading, setLoading] = useState(false)

  const handleAsk = async (e) => {
    e.preventDefault()
    if (!query.trim()) return

    // User ka sawal chat me add kar rahe hain
    const newChat = [...chatLog, { role: 'user', text: query }]
    setChatLog(newChat)
    setQuery('')
    setLoading(true)

    try {
      // 🛑 YAHAN DHYAN DE: Ye API call abhi fail hogi kyunki apna FastAPI backend abhi bana nahi hai!
      const response = await fetch('https://vidrag.onrender.com/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: query })
      })
      
      const data = await response.json()
      
      // Bot ka answer aur links update karo
      setChatLog([...newChat, { 
        role: 'bot', 
        text: data.answer, 
        links: data.links 
      }])
    } catch (error) {
      setChatLog([...newChat, { 
        role: 'bot', 
        text: "❌ Backend abhi start nahi hua hai bhai! Pehle Python (FastAPI) server banayenge tabhi ye engine chalega." 
      }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white font-sans pb-24">
      {/* Header */}
      <header className="p-4 bg-gray-800 text-center shadow-lg border-b border-gray-700">
        <h1 className="text-3xl font-bold text-blue-400">DSA AI Assistant 🤖</h1>
        <p className="text-gray-400 text-sm mt-1">Pratyush bhai ki videos se powered!</p>
      </header>

      {/* Chat Space */}
      <div className="max-w-4xl mx-auto p-4 flex flex-col space-y-6 mt-4">
        {chatLog.length === 0 ? (
          <div className="text-center text-gray-500 mt-20">
            <p className="text-xl">Bhai, tera frontend ready hai.</p>
            <p>Neeche apna sawal type kar aur UI ka look check kar!</p>
          </div>
        ) : (
          chatLog.map((msg, index) => (
            <div key={index} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`p-4 max-w-[85%] rounded-xl shadow-md ${msg.role === 'user' ? 'bg-blue-600 rounded-br-none' : 'bg-gray-800 border border-gray-700 rounded-bl-none'}`}>
                <p className="whitespace-pre-wrap leading-relaxed">{msg.text}</p>
                
                {/* YouTube Links ka section */}
                {msg.links && msg.links.length > 0 && (
                  <div className="mt-4 pt-3 border-t border-gray-600">
                    <p className="text-sm text-gray-400 mb-2 font-semibold">📺 Video References:</p>
                    <ul className="space-y-2">
                      {msg.links.map((link, i) => (
                        <li key={i}>
                          <a href={link} target="_blank" rel="noreferrer" className="flex items-center text-blue-400 hover:text-blue-300 hover:underline text-sm bg-gray-700 p-2 rounded transition-colors">
                            👉 Exact time par video dekhne ke liye click kar
                          </a>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          ))
        )}
        
        {/* Loading Animation */}
        {loading && (
          <div className="flex justify-start">
            <div className="p-4 rounded-xl bg-gray-800 border border-gray-700 text-gray-400 rounded-bl-none flex items-center space-x-2">
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></div>
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce delay-75"></div>
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce delay-150"></div>
              <span className="ml-2">Soch raha hu bhai...</span>
            </div>
          </div>
        )}
      </div>

      {/* Input Box - Ekdum bottom me fixed */}
      <div className="fixed bottom-0 left-0 w-full bg-gray-800 p-4 border-t border-gray-700 shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.1)]">
        <form onSubmit={handleAsk} className="max-w-4xl mx-auto flex gap-3">
          <input 
            type="text" 
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="DP aur Greedy approach me kya farak hai?"
            className="flex-1 bg-gray-700 text-white px-4 py-3 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 transition-shadow"
          />
          <button 
            type="submit" 
            disabled={loading}
            className="bg-blue-600 hover:bg-blue-500 px-8 py-3 rounded-xl font-bold transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center shadow-lg"
          >
            Send 🚀
          </button>
        </form>
      </div>
    </div>
  )
}

export default App