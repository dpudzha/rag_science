import { useState } from 'react'
import Sidebar from './components/Sidebar'
import ChatPanel from './components/ChatPanel'
import IngestPanel from './components/IngestPanel'
import ConfigPanel from './components/ConfigPanel'
import './App.css'

type Tab = 'chat' | 'ingest' | 'config'

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('chat')

  return (
    <div className="app">
      <Sidebar activeTab={activeTab} onTabChange={setActiveTab} />
      <main className="main-content">
        {activeTab === 'chat' && <ChatPanel />}
        {activeTab === 'ingest' && <IngestPanel />}
        {activeTab === 'config' && <ConfigPanel />}
      </main>
    </div>
  )
}

export default App
