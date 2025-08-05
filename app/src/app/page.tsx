"use client";

import Output from "@/components/Output";
import TextArea from "@/components/TextArea";
import ThemeToggle from "@/components/ThemeToggle";
import { type ChatOutput } from "@/types";
import { useState, useEffect } from "react";

// Logging utility for main page
const log = {
  info: (message: string, data?: any) => {
    console.log(`🏠 [HomePage] ${message}`, data ? data : "");
  },
  debug: (message: string, data?: any) => {
    console.debug(`🟡 [HomePage] ${message}`, data ? data : "");
  },
};

export default function Home() {
  const [outputs, setOutputs] = useState<ChatOutput[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);

  useEffect(() => {
    log.info("HomePage mounted");
  }, []);

  useEffect(() => {
    log.debug(`Outputs updated: ${outputs.length} total conversations`);
  }, [outputs]);

  useEffect(() => {
    log.debug(`Generation state: ${isGenerating ? "ACTIVE" : "IDLE"}`);
  }, [isGenerating]);

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm border-b border-white/20 dark:border-gray-700/50 shadow-sm">
        <div className="max-w-4xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
                <svg
                  className="w-6 h-6 text-white"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                  />
                </svg>
              </div>
              <div>
                <h1 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  Chatbot
                </h1>
                <p className="text-sm text-gray-600 dark:text-gray-300">
                  Powered by OpenAI & SerpAPI
                </p>
              </div>
            </div>
            <ThemeToggle />
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        <div className="max-w-4xl mx-auto w-full px-6 py-8 flex-1">
          {outputs.length === 0 ? (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center space-y-6">
                <div className="w-20 h-20 bg-gradient-to-r from-blue-500 to-purple-600 dark:from-cyan-400 dark:to-purple-500 rounded-full flex items-center justify-center mx-auto mb-6">
                  <svg
                    className="w-10 h-10 text-white"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
                    />
                  </svg>
                </div>
                <h2 className="text-4xl font-bold bg-gradient-to-r from-gray-900 to-gray-600 dark:from-white dark:to-gray-300 bg-clip-text text-transparent">
                  What do you want to know?
                </h2>
                <p className="text-lg text-gray-600 dark:text-gray-300 max-w-md mx-auto">
                  Ask me anything! I can help with research, calculations, and
                  provide detailed answers using web search.
                </p>
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              {outputs.map((output, i) => (
                <Output key={i} output={output} />
              ))}
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm border-t border-white/20 dark:border-gray-700/50 shadow-lg">
          <div className="max-w-4xl mx-auto px-6 py-4">
            <TextArea
              setIsGenerating={setIsGenerating}
              isGenerating={isGenerating}
              outputs={outputs}
              setOutputs={setOutputs}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
