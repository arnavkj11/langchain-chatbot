"use client";

import { useEffect, useRef, useState } from "react";
import { IncompleteJsonParser } from "incomplete-json-parser";
import { ChatOutput } from "@/types";

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
  // Parser instances to handle incomplete JSON streaming responses
  const parser = new IncompleteJsonParser();
  const stepParser = new IncompleteJsonParser();

  const [text, setText] = useState("");
  const textAreaRef = useRef<HTMLTextAreaElement>(null);

  // Handles form submission
  async function submit(e: React.FormEvent) {
    e.preventDefault();
    sendMessage(text);
    setText("");
  }

  // Sends message to the api and handles streaming response processing
  const sendMessage = async (text: string) => {
    console.log("🚀 Sending message:", text);

    // Reset parsers for new message
    parser.reset();
    stepParser.reset();

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

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
      console.log("📡 API URL:", `${apiUrl}/invoke`);

      // Send as URL-encoded form data (the way FastAPI expects it)
      const formData = new URLSearchParams();
      formData.append("content", text);

      console.log("📤 Making API request with content:", text);
      const res = await fetch(`${apiUrl}/invoke`, {
        method: "POST",
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body: formData,
      });

      console.log("📥 Response status:", res.status);
      console.log(
        "📥 Response headers:",
        Object.fromEntries(res.headers.entries())
      );

      if (!res.ok) {
        const errorText = await res.text();
        console.error("❌ API Error:", res.status, errorText);
        throw new Error(`API Error: ${res.status} - ${errorText}`);
      }

      const data = res.body;
      if (!data) {
        console.error("❌ No response body");
        setIsGenerating(false);
        return;
      }

      console.log("✅ Starting to read stream...");
      const reader = data.getReader();
      const decoder = new TextDecoder();
      let done = false;
      let answer = { answer: "", tools_used: [] };
      let currentSteps: { name: string; result: Record<string, string> }[] = [];
      let buffer = "";
      let chunkCount = 0;

      // Process streaming response chunks and parse steps/results
      while (!done) {
        try {
          const { value, done: doneReading } = await reader.read();
          done = doneReading;
          let chunkValue = decoder.decode(value);
          chunkCount++;

          if (chunkCount <= 5) {
            // Log first 5 chunks for debugging
            console.log(`📦 Chunk ${chunkCount}:`, chunkValue);
          }

          if (!chunkValue) continue;

          buffer += chunkValue;

          // Handle different types of steps in the response stream - regular steps and final answer
          if (buffer.includes("</step_name>")) {
            const stepNameMatch = buffer.match(
              /<step_name>([^<]*)<\/step_name>/
            );
            if (stepNameMatch) {
              const [_, stepName] = stepNameMatch;
              console.log("🔧 Processing step:", stepName);

              try {
                if (stepName !== "final_answer") {
                  const fullStepPattern =
                    /<step><step_name>([^<]*)<\/step_name>([^<]*?)(?=<step>|<\/step>|$)/g;
                  const matches = [...buffer.matchAll(fullStepPattern)];

                  for (const match of matches) {
                    const [fullMatch, matchStepName, jsonStr] = match;
                    if (jsonStr) {
                      try {
                        // Use incomplete JSON parser for step results to handle partial JSON
                        stepParser.write(jsonStr);
                        const parsedResults = stepParser.getObjects();
                        
                        if (parsedResults && typeof parsedResults === 'object') {
                          console.log(
                            "✅ Parsed step result:",
                            matchStepName,
                            parsedResults
                          );
                          currentSteps.push({ name: matchStepName, result: parsedResults });
                          buffer = buffer.replace(fullMatch, "");
                        }
                        stepParser.reset();
                      } catch (error) {
                        console.error("❌ Failed to parse step JSON:", error);
                        stepParser.reset(); // Reset parser state on error
                      }
                    }
                  }
                } else {
                  // If it's the final answer step, parse the streaming JSON using incomplete-json-parser
                  const jsonMatch = buffer.match(
                    /(?<=<step><step_name>final_answer<\/step_name>)(.*)/
                  );
                  if (jsonMatch) {
                    const [_, jsonStr] = jsonMatch;
                    console.log("🎯 Processing final answer:", jsonStr);
                    try {
                      parser.write(jsonStr);
                      const result = parser.getObjects();
                      if (result && typeof result === 'object') {
                        answer = result;
                        console.log("✅ Final answer parsed:", answer);
                      }
                    } catch (error) {
                      console.error("❌ Failed to parse final answer JSON:", error);
                      parser.reset(); // Reset parser state on error
                    }
                  }
                }
              } catch (e) {
                console.error("❌ Failed to parse step:", e);
              }
            }
          }

          // Update output with current content and steps
          setOutputs((prevState) => {
            const lastOutput = prevState[prevState.length - 1];
            return [
              ...prevState.slice(0, -1),
              {
                ...lastOutput,
                steps: currentSteps,
                result: answer,
              },
            ];
          });
        } catch (streamError) {
          console.error("❌ Stream processing error:", streamError);
          break;
        }
      }

      console.log("✅ Stream processing completed");
      console.log("📊 Final steps:", currentSteps);
      console.log("📊 Final answer:", answer);
    } catch (error) {
      console.error("❌ API call failed:", error);
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      setOutputs((prevState) => {
        const lastOutput = prevState[prevState.length - 1];
        return [
          ...prevState.slice(0, -1),
          {
            ...lastOutput,
            result: { answer: `Error: ${errorMessage}`, tools_used: [] },
          },
        ];
      });
    } finally {
      setIsGenerating(false);
      // Reset parsers to clean up state
      parser.reset();
      stepParser.reset();
      console.log("🏁 Message processing finished");
    }
  };

  // Submit form when Enter is pressed (without Shift)
  function submitOnEnter(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.code === "Enter" && !e.shiftKey) {
      submit(e);
    }
  }

  // Dynamically adjust textarea height based on content
  const adjustHeight = () => {
    const textArea = textAreaRef.current;
    if (textArea) {
      textArea.style.height = "auto";
      textArea.style.height = `${textArea.scrollHeight}px`;
    }
  };

  // Adjust height whenever text content changes
  useEffect(() => {
    adjustHeight();
  }, [text]);

  // Add resize event listener to adjust height on window resize
  useEffect(() => {
    const handleResize = () => adjustHeight();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

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
