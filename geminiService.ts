
import { GoogleGenAI, Type } from "@google/genai";
import { UserPreferences, StudyNotes } from "./types";

const SYSTEM_PROMPT = `
  You are an expert B.Pharm professor and academic mentor specializing in Indian PCI (Pharmacy Council of India) syllabus.
  Your task is to generate highly structured, exam-oriented study notes for first-year pharmacy students.
  
  CRITICAL RULES:
  1. Language: Simple, academic, and professional.
  2. Context: Suitable for 2, 5, and 10 marks university questions.
  3. Content: Must be technically accurate and include pharmacy-specific examples.
  4. Mnemonics: If requested, provide catchy and easy-to-remember mnemonics for classifications or lists.
  5. Diagrams: Provide clear, text-based descriptions of labeled diagrams that a student can draw.
  
  You must return the response as a valid JSON object matching the provided schema exactly.
`;

const getApiKey = () => {
  const key = process.env.API_KEY;
  if (!key) {
    console.warn("API Key is missing. Ensure process.env.API_KEY is configured.");
  }
  return key || "";
};

export async function generateStudyNotes(prefs: UserPreferences): Promise<StudyNotes> {
  const ai = new GoogleGenAI({ apiKey: getApiKey() });
  
  const userPrompt = `
    Subject: ${prefs.subject}
    Semester: ${prefs.semester}
    Topic: ${prefs.topic}
    Level of Detail: ${prefs.length}
    Target University: ${prefs.university || 'Any PCI-affiliated University'}
    
    Specific Requirements:
    - Include Diagram Description: ${prefs.includeDiagrams}
    - Include Mnemonics: ${prefs.includeMnemonics}
    - Include Clinical/Practical Correlation: ${prefs.includeClinicalCorrelation}
  `;

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-3-pro-preview',
      contents: [{ parts: [{ text: userPrompt }] }],
      config: {
        systemInstruction: SYSTEM_PROMPT,
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            introduction: { type: Type.STRING },
            definition: { type: Type.STRING },
            classification: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  type: { type: Type.STRING },
                  explanation: { type: Type.STRING }
                },
                required: ["type", "explanation"]
              }
            },
            detailedExplanation: { type: Type.ARRAY, items: { type: Type.STRING } },
            examples: { type: Type.ARRAY, items: { type: Type.STRING } },
            diagramDescription: { type: Type.STRING },
            examPoints: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  point: { type: Type.STRING },
                  mnemonic: { type: Type.STRING }
                },
                required: ["point"]
              }
            },
            shortAnswerQuestions: { type: Type.ARRAY, items: { type: Type.STRING } },
            longAnswerQuestions: { type: Type.ARRAY, items: { type: Type.STRING } },
            pyqs: { type: Type.ARRAY, items: { type: Type.STRING } },
            vivaQuestions: { type: Type.ARRAY, items: { type: Type.STRING } },
            clinicalCorrelation: { type: Type.STRING }
          },
          required: ["introduction", "detailedExplanation", "examples", "examPoints", "shortAnswerQuestions", "longAnswerQuestions", "pyqs", "vivaQuestions"]
        }
      }
    });

    const text = response.text;
    if (!text) throw new Error("The AI returned an empty response. Please try again.");
    return JSON.parse(text) as StudyNotes;
  } catch (error: any) {
    console.error("Gemini API Error:", error);
    throw new Error(error.message || "Could not generate academic notes.");
  }
}

export async function analyzePharmacyImage(base64Image: string, mimeType: string): Promise<string> {
  const ai = new GoogleGenAI({ apiKey: getApiKey() });
  
  const prompt = `
    Analyze this pharmacy academic image. IDENTIFY correctly and PROVIDE academic context.
    
    Structure your response as follows:
    # Identification
    [Clear name of what is shown]

    # Academic Context
    [Which B.Pharm subject/semester does this belong to?]

    # Exam Focus
    [Important points for a 5-mark answer]

    # Practical & Safety
    [Laboratory precautions or clinical relevance]

    # Correction Note (if applicable)
    [Point out any spelling/factual errors in the image if it's a student's note]
    
    Use markdown format with # for headings (they will be cleaned by UI) and * bullet points. Be professional and highly detailed.
  `;

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-3-pro-preview',
      contents: {
        parts: [
          { inlineData: { data: base64Image, mimeType } },
          { text: prompt }
        ]
      },
      config: {
        systemInstruction: "You are an expert Pharmacy Professor. Your analysis is used by B.Pharm students for exam preparation. Be technically rigorous and structured."
      }
    });

    return response.text || "Analysis unavailable for this image.";
  } catch (error: any) {
    console.error("Image Analysis Error:", error);
    throw new Error("Failed to process image lab analysis.");
  }
}

export function createPharmacyChatSession(): any {
  const ai = new GoogleGenAI({ apiKey: getApiKey() });
  
  return ai.chats.create({
    model: 'gemini-3-pro-preview',
    config: {
      systemInstruction: "You are 'PharmAssistant', a virtual mentor for B.Pharm students. Provide helpful, exam-focused answers. Always advise consulting a professional for health issues."
    }
  });
}