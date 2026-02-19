import './Sidebar.css'

type Tab = 'chat' | 'ingest' | 'config'

interface SidebarProps {
  activeTab: Tab;
  onTabChange: (tab: Tab) => void;
}

const navItems: { tab: Tab; label: string; icon: string }[] = [
  { tab: 'chat', label: 'Chat', icon: '\u{1F4AC}' },
  { tab: 'ingest', label: 'Ingest', icon: '\u{1F4C4}' },
  { tab: 'config', label: 'Config', icon: '\u{2699}' },
]

function Sidebar({ activeTab, onTabChange }: SidebarProps) {
  return (
    <aside className="sidebar">
      <div className="sidebar-title">RAG Science</div>
      <nav className="sidebar-nav">
        {navItems.map(({ tab, label, icon }) => (
          <button
            key={tab}
            className={`sidebar-item ${activeTab === tab ? 'active' : ''}`}
            onClick={() => onTabChange(tab)}
          >
            <span className="sidebar-icon">{icon}</span>
            {label}
          </button>
        ))}
      </nav>
    </aside>
  )
}

export default Sidebar
