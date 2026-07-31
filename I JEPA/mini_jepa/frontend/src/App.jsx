import React, { useState } from 'react'

export default function App() {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const onFileChange = (e) => setFile(e.target.files[0])

  const upload = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)
    const form = new FormData()
    form.append('image', file)

    try {
      const resp = await fetch('/api/predict', { method: 'POST', body: form })
      if (!resp.ok) throw new Error(await resp.text())
      const json = await resp.json()
      setResult(json)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <h1>MiniJEPA Demo</h1>
      <p>Upload a masked image with a missing square.</p>

      <input type="file" accept="image/*" onChange={onFileChange} />
      <button onClick={upload} disabled={loading || !file}>
        {loading ? 'Running...' : 'Run MiniJEPA'}
      </button>

      {error && <div className="error">Error: {error}</div>}

      {result && (
        <div className="results">
          <div>
            <h3>Completed</h3>
            <img src={result.completed_image} alt="completed" />
          </div>
          <div>
            <h3>Panel</h3>
            <img src={result.panel_image} alt="panel" />
          </div>
        </div>
      )}

      <footer>
        <small>Backend: POST /api/predict — runs lazy-loaded model on first use.</small>
      </footer>
    </div>
  )
}
