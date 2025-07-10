"use client";

import { useState, useEffect } from "react";

export default function Home() {
  const [prompt, setPrompt] = useState("");
  const [image, setImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) {
      setError("Please enter a prompt");
      return;
    }
    
    setLoading(true);
    setError(null);
    setImage(null);
    
    try {
      console.log("Generating image for prompt:", prompt);
      const res = await fetch("http://localhost:5001/generate_fast", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });
      
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP ${res.status}: ${res.statusText}`);
      }
      
      const data = await res.json();
      console.log("Response data:", data);
      
      if (data.image_path) {
        setImage(data.image_path);
      } else {
        throw new Error("No image path in response");
      }
    } catch (err: any) {
      console.error("Generation error:", err);
      setError(err.message || "Failed to generate image. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  if (!mounted) {
    return (
      <div style={{ maxWidth: 600, margin: "2rem auto", padding: 24 }}>
        <h1>AI Image Generator</h1>
        <p>Loading...</p>
      </div>
    );
  }

  return (
    <div style={{ maxWidth: 600, margin: "2rem auto", padding: 24 }}>
      <h1>🎨 AI Image Generator</h1>
      <p style={{ color: "#666", marginBottom: 24 }}>
        Enter a description and generate beautiful AI images instantly!
      </p>
      
      <form onSubmit={handleSubmit} style={{ marginBottom: 24 }}>
        <input
          type="text"
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
          placeholder="Enter your prompt (e.g., 'A beautiful forest with sunlight')"
          style={{ 
            width: "70%", 
            padding: 12, 
            fontSize: 16, 
            border: "1px solid #ddd",
            borderRadius: 4
          }}
          required
          disabled={loading}
        />
        <button
          type="submit"
          style={{ 
            marginLeft: 12, 
            padding: "12px 24px", 
            fontSize: 16,
            backgroundColor: loading ? "#ccc" : "#007bff",
            color: "white",
            border: "none",
            borderRadius: 4,
            cursor: loading ? "not-allowed" : "pointer"
          }}
          disabled={loading}
        >
          {loading ? "🔄 Generating..." : "✨ Generate"}
        </button>
      </form>
      
      {error && (
        <div style={{ 
          color: "red", 
          marginBottom: 16, 
          padding: 12, 
          backgroundColor: "#ffe6e6",
          border: "1px solid #ffcccc",
          borderRadius: 4
        }}>
          ❌ {error}
        </div>
      )}
      
      {loading && (
        <div style={{ 
          textAlign: "center", 
          marginBottom: 16,
          padding: 20,
          backgroundColor: "#f8f9fa",
          borderRadius: 4
        }}>
          <p>🎨 Creating your image... This may take a minute.</p>
        </div>
      )}
      
      {image && (
        <div style={{ textAlign: "center" }}>
          <h3>✨ Your Generated Image</h3>
          <img 
            src={`http://localhost:5001/${image}`} 
            alt="Generated" 
            style={{ 
              maxWidth: "100%", 
              borderRadius: 8,
              boxShadow: "0 4px 8px rgba(0,0,0,0.1)"
            }} 
          />
          <p style={{ marginTop: 12, color: "#666" }}>
            Prompt: "{prompt}"
          </p>
        </div>
      )}
    </div>
  );
}
