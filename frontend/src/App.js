// src/App.js
import React, { useState } from 'react';
import axios from 'axios';
import { Send, Loader2, ThumbsUp, ThumbsDown, BarChart3 } from 'lucide-react';
import './App.css';

function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

  const analyzeText = async () => {
    if (!text.trim()) {
      setError('Please enter some text');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await axios.post(`${API_URL}/predict`, {
        text: text
      });

      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.message || 'Failed to analyze sentiment');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      analyzeText();
    }
  };

  const examples = [
    "This movie was absolutely brilliant! The acting was superb.",
    "Worst movie I've ever seen. Complete waste of time.",
    "It was okay, nothing special but not terrible either."
  ];

  return (
    <div className="App">
      <div className="container">
        <header className="header">
          <h1>🎭 Sentiment Analyzer</h1>
          <p>AI-powered sentiment analysis using NLP</p>
        </header>

        <div className="input-section">
          <textarea
            className="text-input"
            placeholder="Enter your text here... (e.g., movie review, tweet, product review)"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyPress={handleKeyPress}
            rows={6}
          />
          
          <button 
            className="analyze-btn"
            onClick={analyzeText}
            disabled={loading || !text.trim()}
          >
            {loading ? (
              <>
                <Loader2 className="icon spinning" />
                Analyzing...
              </>
            ) : (
              <>
                <Send className="icon" />
                Analyze Sentiment
              </>
            )}
          </button>

          {error && (
            <div className="error-message">
              ⚠️ {error}
            </div>
          )}
        </div>

        {result && (
          <div className="results">
            <div className={`sentiment-badge ${result.sentiment}`}>
              {result.sentiment === 'positive' ? (
                <><ThumbsUp className="icon" /> Positive</>
              ) : (
                <><ThumbsDown className="icon" /> Negative</>
              )}
            </div>

            <div className="confidence">
              <BarChart3 className="icon" />
              <span>Confidence: {result.confidence}%</span>
            </div>

            <div className="scores">
              <div className="score-bar">
                <div className="score-label">
                  <span>😊 Positive</span>
                  <span>{result.scores.positive}%</span>
                </div>
                <div className="progress-bar">
                  <div 
                    className="progress positive"
                    style={{ width: `${result.scores.positive}%` }}
                  />
                </div>
              </div>

              <div className="score-bar">
                <div className="score-label">
                  <span>😞 Negative</span>
                  <span>{result.scores.negative}%</span>
                </div>
                <div className="progress-bar">
                  <div 
                    className="progress negative"
                    style={{ width: `${result.scores.negative}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        )}

        <div className="examples">
          <h3>💡 Try these examples:</h3>
          <div className="example-buttons">
            {examples.map((example, idx) => (
              <button
                key={idx}
                className="example-btn"
                onClick={() => setText(example)}
              >
                {example.substring(0, 50)}...
              </button>
            ))}
          </div>
        </div>

        <footer className="footer">
          <p>Built with React + Flask + Scikit-learn</p>
          <p>88.95% accuracy on 50,000 IMDB reviews</p>
        </footer>
      </div>
    </div>
  );
}

export default App;