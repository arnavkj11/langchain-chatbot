"use client";

import { useEffect, useRef, useState } from "react";
import { IncompleteJsonParser } from "incomplete-json-parser";
import { ChatOutput } from "@/types";

// Logging utility functions
const log = {
  info: (message: string, data?: any) => {
    console.log(`🔵 [TextArea] ${message}`, data ? data : '');
  },
  error: (message: string, error?: any) => {
    console.error(`🔴 [TextArea] ${message}`, error ? error : '');
  },
  debug: (message: string, data?: any) => {
    console.debug(`🟡 [TextArea] ${message}`, data ? data : '');
  },
  success: (message: string, data?: any) => {
    console.log(`🟢 [TextArea] ${message}`, data ? data : '');
  }
};

const TextArea = ({
  setIsGenerating,
  isGenerating,
  setOutputs,
  outputs,
}: {
  setIsGenerating: React.Dispatch<React.SetStateAction<boolean>>;
  isGenerating: boolean;
  setOutputs: React.Dispatch<React.SetStateAction<ChatOutput[]>>;
  outputs: ChatOutput[];
}) => {
  // Parser instance to handle incomplete JSON streaming responses
  const parser = new IncompleteJsonParser();

  const [text, setText] = useState("");
  const textAreaRef = useRef<HTMLTextAreaElement>(null);

  // Handles form submission
  async function submit(e: React.FormEvent) {
    e.preventDefault();
    log.info(`Form submission triggered with text: "${text}"`);
    sendMessage(text);
    setText("");
  }

  // Sends message to the api and handles streaming response processing
  const sendMessage = async (text: string) => {
    const requestId = Math.random().toString(36).substr(2, 8);
    log.info(`[${requestId}] Starting message send process`, { 
      messageLength: text.length, 
      currentOutputsCount: outputs.length 
    });

    const newOutputs = [
      ...outputs,
      {
        question: text,
        steps: [],
        result: {
          answer: "",
          tools_used: [],
        },
      },
    ];

    setOutputs(newOutputs);
    setIsGenerating(true);
    log.debug(`[${requestId}] UI state updated - generating: true, outputs count: ${newOutputs.length}`);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      log.info(`[${requestId}] Making API request to: ${apiUrl}/invoke`);
      
      const startTime = performance.now();
      const res = await fetch(`${apiUrl}/invoke?content=${text}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(text),
      });

      const connectionTime = performance.now() - startTime;
      log.info(`[${requestId}] API connection established in ${connectionTime.toFixed(2)}ms`);

      if (!res.ok) {
        log.error(`[${requestId}] API request failed with status: ${res.status} ${res.statusText}`);
        throw new Error("Error");
      }

      const data = res.body;
      if (!data) {
        log.error(`[${requestId}] No response body received`);
        setIsGenerating(false);
        return;
      }

      log.success(`[${requestId}] Starting stream processing`);
      const reader = data.getReader();
      const decoder = new TextDecoder();
      let done = false;
      let answer = { answer: "", tools_used: [] };
      let currentSteps: { name: string; result: Record<string, string> }[] = [];
      let buffer = "";
      let chunkCount = 0;
      let stepCount = 0;

      // Process streaming response chunks and parse steps/results
      while (!done) {
        const { value, done: doneReading } = await reader.read();
        done = doneReading;
        let chunkValue = decoder.decode(value);
        chunkCount++;
        
        log.debug(`[${requestId}] Chunk ${chunkCount} received`, { 
          length: chunkValue?.length || 0,
          preview: chunkValue?.substring(0, 50) + (chunkValue?.length > 50 ? '...' : '')
        });
        if (!chunkValue) continue;

        buffer += chunkValue;

        // Handle different types of steps in the response stream - regular steps and final answer
        if (buffer.includes("</step_name>")) {
          const stepNameMatch = buffer.match(/<step_name>([^<]*)<\/step_name>/);
          if (stepNameMatch) {
            const [_, stepName] = stepNameMatch;
            log.info(`[${requestId}] Processing step: ${stepName}`);
            
            try {
              if (stepName !== "final_answer") {
                stepCount++;
                log.debug(`[${requestId}] Processing regular step ${stepCount}: ${stepName}`);
                
                const fullStepPattern =
                  /<step><step_name>([^<]*)<\/step_name>([^<]*?)(?=<step>|<\/step>|$)/g;
                const matches = [...buffer.matchAll(fullStepPattern)];

                for (const match of matches) {
                  const [fullMatch, matchStepName, jsonStr] = match;
                  if (jsonStr) {
                    try {
                      const result = JSON.parse(jsonStr);
                      currentSteps.push({ name: matchStepName, result });
                      buffer = buffer.replace(fullMatch, "");
                      log.success(`[${requestId}] Step ${matchStepName} parsed successfully`, result);
                    } catch (parseError) {
                      log.error(`[${requestId}] Failed to parse step JSON for ${matchStepName}`, parseError);
                    }
                  }
                }
              } else {
                log.info(`[${requestId}] Processing final answer step`);
                // If it's the final answer step, parse the streaming JSON using incomplete-json-parser
                const jsonMatch = buffer.match(
                  /(?<=<step><step_name>final_answer<\/step_name>)(.*)/
                );
                if (jsonMatch) {
                  const [_, jsonStr] = jsonMatch;
                  parser.write(jsonStr);
                  const result = parser.getObjects();
                  answer = result;
                  parser.reset();
                  log.success(`[${requestId}] Final answer parsed`, { 
                    answerLength: answer.answer?.length || 0,
                    toolsUsed: answer.tools_used 
                  });
                }
              }
            } catch (e) {
              log.error(`[${requestId}] Failed to parse step:`, e);
            }
          }
        }

        // Update output with current content and steps
        setOutputs((prevState) => {
          const lastOutput = prevState[prevState.length - 1];
          const updated = [
            ...prevState.slice(0, -1),
            {
              ...lastOutput,
              steps: currentSteps,
              result: answer,
            },
          ];
          
          log.debug(`[${requestId}] UI updated`, { 
            stepsCount: currentSteps.length,
            hasAnswer: !!answer.answer 
          });
          
          return updated;
        });
      }
      
      const totalTime = performance.now() - startTime;
      log.success(`[${requestId}] Stream processing completed`, {
        totalTime: `${totalTime.toFixed(2)}ms`,
        chunksProcessed: chunkCount,
        stepsProcessed: stepCount,
        finalAnswerLength: answer.answer?.length || 0
      });
      
    } catch (error) {
      log.error(`[${requestId}] Error in sendMessage:`, error);
    } finally {
      setIsGenerating(false);
      log.info(`[${requestId}] Message processing completed - generating: false`);
    }
  };

  // Submit form when Enter is pressed (without Shift)
  function submitOnEnter(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.code === "Enter" && !e.shiftKey) {
      log.debug("Enter key pressed - submitting form");
      submit(e);
    }
  }

  // Dynamically adjust textarea height based on content
  const adjustHeight = () => {
    const textArea = textAreaRef.current;
    if (textArea) {
      const oldHeight = textArea.style.height;
      textArea.style.height = "auto";
      textArea.style.height = `${textArea.scrollHeight}px`;
      
      if (oldHeight !== textArea.style.height) {
        log.debug(`Textarea height adjusted to ${textArea.style.height}`);
      }
    }
  };

  // Adjust height whenever text content changes
  useEffect(() => {
    adjustHeight();
  }, [text]);

  // Add resize event listener to adjust height on window resize
  useEffect(() => {
    log.info("TextArea component mounted");
    const handleResize = () => {
      log.debug("Window resized - adjusting textarea height");
      adjustHeight();
    };
    window.addEventListener("resize", handleResize);
    return () => {
      log.debug("TextArea component unmounting");
      window.removeEventListener("resize", handleResize);
    };
  }, []);

  // Log when generating state changes
  useEffect(() => {
    log.info(`Generation state changed: ${isGenerating ? 'STARTED' : 'STOPPED'}`);
  }, [isGenerating]);

  // Log when text changes
  useEffect(() => {
    if (text.length > 0) {
      log.debug(`Text input changed: ${text.length} characters`);
    }
  }, [text]);

  return (
    <form onSubmit={submit} className="w-full">
      <div className="relative">
        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-lg border border-gray-200 dark:border-gray-600 hover:shadow-xl transition-shadow duration-200">
          <textarea
            ref={textAreaRef}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => submitOnEnter(e)}
            rows={1}
            className="w-full p-4 pr-14 bg-transparent min-h-[60px] max-h-32 focus:outline-none resize-none text-gray-800 dark:text-gray-200 placeholder-gray-500 dark:placeholder-gray-400 rounded-2xl"
            placeholder="Ask me anything..."
            disabled={isGenerating}
          />
          <button
            type="submit"
            disabled={isGenerating || !text.trim()}
            className="absolute right-3 top-1/2 transform -translate-y-1/2 disabled:opacity-50 disabled:cursor-not-allowed bg-gradient-to-r from-blue-500 to-purple-600 dark:from-cyan-400 dark:to-purple-500 hover:from-blue-600 hover:to-purple-700 dark:hover:from-cyan-500 dark:hover:to-purple-600 transition-all duration-200 w-10 h-10 rounded-full flex items-center justify-center group shadow-lg"
          >
            {isGenerating ? (
              <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
            ) : (
              <ArrowIcon />
            )}
          </button>
        </div>

        {/* Gradient border effect */}
        <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500 to-purple-600 dark:from-cyan-400 dark:to-purple-500 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-200 -z-10" />
      </div>
    </form>
  );
};

const ArrowIcon = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="18"
    height="18"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2.5"
    strokeLinecap="round"
    strokeLinejoin="round"
    className="text-white group-hover:translate-x-0.5 transition-transform duration-200"
  >
    <path d="M5 12h14" />
    <path d="m12 5 7 7-7 7" />
  </svg>
);

export default TextArea;
