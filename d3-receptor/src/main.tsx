import React, { Suspense } from 'react'
import ReactDOM from 'react-dom/client'
import './index.css'

// Use React.lazy for code splitting
const App = React.lazy(() => import('./App'))

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <Suspense fallback={<div className="loading-app">Loading application...</div>}>
      <App />
    </Suspense>
  </React.StrictMode>,
)
