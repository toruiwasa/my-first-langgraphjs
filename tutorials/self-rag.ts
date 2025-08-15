import "dotenv/config";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { BedrockEmbeddings } from "@langchain/aws";

const urls = [
  "https://lilianweng.github.io/posts/2023-06-23-agent/",
  "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
  "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
];

const docs = await Promise.all(
  urls.map((url) => new CheerioWebBaseLoader(url).load())
);
const docsList = docs.flat();

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 250,
});
const docSplits = await textSplitter.splitDocuments(docsList);

// Add to vectorDB
const embeddings = new BedrockEmbeddings({
  model: "amazon.titan-embed-text-v2:0",
  region: process.env.BEDROCK_AWS_REGION ?? "us-east-1",
  credentials: {
    secretAccessKey: process.env.BEDROCK_AWS_SECRET_ACCESS_KEY ?? "",
    accessKeyId: process.env.BEDROCK_AWS_ACCESS_KEY_ID ?? "",
  },
});
const vectorStore = await MemoryVectorStore.fromDocuments(
  docSplits,
  embeddings
);
const retriever = vectorStore.asRetriever();

import { Annotation } from "@langchain/langgraph";
import { type DocumentInterface } from "@langchain/core/documents";

// Represents the state of our graph.
const GraphState = Annotation.Root({
  documents: Annotation<DocumentInterface[]>({
    reducer: (x, y) => y ?? x ?? [],
  }),
  question: Annotation<string>({
    reducer: (x, y) => y ?? x ?? "",
  }),
  generation: Annotation<string>({
    reducer: (x, y) => y ?? x,
    default: () => "",
  }),
  generationVQuestionGrade: Annotation<string>({
    reducer: (x, y) => y ?? x,
  }),
  generationVDocumentsGrade: Annotation<string>({
    reducer: (x, y) => y ?? x,
  }),
});

import { z } from "zod";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { pull } from "langchain/hub";
import { ChatBedrockConverse } from "@langchain/aws";
import { StringOutputParser } from "@langchain/core/output_parsers";
import type { RunnableConfig } from "@langchain/core/runnables";
import { formatDocumentsAsString } from "langchain/util/document";

// Defnine the LLM once. We'll reuse it throughout the graph.
const model = new ChatBedrockConverse({
  model: "us.amazon.nova-micro-v1:0",
  region: process.env.BEDROCK_AWS_REGION ?? "us-east-1",
  credentials: {
    secretAccessKey: process.env.BEDROCK_AWS_SECRET_ACCESS_KEY ?? "",
    accessKeyId: process.env.BEDROCK_AWS_ACCESS_KEY_ID ?? "",
  },
  temperature: 0,
});

/**
 * Retrieve documents
 *
 * @param {typeof GraphState.State} state The current state of the graph.
 * @param {RunnableConfig | undefined} config The configuration object for tracing.
 * @returns {Promise<Partial<typeof GraphState.State>>} The new state object.
 */
async function retrieve(
  state: typeof GraphState.State,
  config?: RunnableConfig
): Promise<Partial<typeof GraphState.State>> {
  console.log("---RETRIEVE---");

  const documents = await retriever
    .withConfig({ runName: "FetchRelevantDocuments" })
    .invoke(state.question, config);

  return {
    documents,
  };
}

/**
 * Generate answer
 *
 * @param {typeof GraphState.State} state The current state of the graph.
 * @returns {Promise<Partial<typeof GraphState.State>>} The new state object.
 */
async function generate(
  state: typeof GraphState.State
): Promise<Partial<typeof GraphState.State>> {
  console.log("---GENERATE---");

  const prompt = await pull<ChatPromptTemplate>("rlm/rag-prompt");
  // Construct the RAG chain by piping the prompt, model, and output parser
  const ragChain = prompt.pipe(model).pipe(new StringOutputParser());

  const generation = await ragChain.invoke({
    context: formatDocumentsAsString(state.documents),
    question: state.question,
  });

  return {
    generation,
  };
}

/**
 * Determines whether the retrieved documents are relevant to the question.
 *
 * @param {typrof GraphState.State} state The current state of the graph.
 * @returns {Promise<Partial<typeof GraphState.State>>} The new state object.
 */
async function gradeDocuments(
  state: typeof GraphState.State
): Promise<Partial<typeof GraphState.State>> {
  console.log("---CHECK RELEVANCE---");

  // pass the name & schema to `withStructuredOutput` which will force the model to call this tool.
  const llmWithTools = model.withStructuredOutput(
    z
      .object({
        binaryScore: z
          .enum(["yes", "no"])
          .describe("Relevance score 'yes' or 'no'"),
      })
      .describe(
        "Grade the relevance of the retrieved documents to the question. Either 'yes' or 'no'"
      ),
    {
      name: "grade",
    }
  );

  const prompt = ChatPromptTemplate.fromTemplate(
    `You are a grader assessing relevance of a retrieved document to a user question.
    Here is the retrieved document:

    {context}

    Here is the user question: {question}

    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.`
  );

  // Chain
  const chain = prompt.pipe(llmWithTools);

  const filteredDocs: Array<DocumentInterface> = [];
  for await (const doc of state.documents) {
    const grade = await chain.invoke({
      context: doc.pageContent,
      question: state.question,
    });
    if (grade.binaryScore === "yes") {
      console.log("---GRADE: DOCUMENT RELEVANT---");
      filteredDocs.push(doc);
    } else {
      console.log("---GRADE: DOCUMENT NOT RELEVANT---");
    }
  }

  return {
    documents: filteredDocs,
  };
}

/**
 * Transform the query to produce a better question.
 *
 * @param {typeof GraphState.State} state The current state of the graph.
 * @returns {Promise<Partial<typeof GraphState.State>>} The new state object.
 */
async function transformQuery(
  state: typeof GraphState.State
): Promise<Partial<typeof GraphState.State>> {
  console.log("---TRANSFORM QUERY---");

  // Pull in the prompt
  const prompt = ChatPromptTemplate.fromTemplate(
    `You are generating a question that is well optimized for semantic search retrieval.
    Look at the input and try to reason about the underlying semantic intent / meaning.

    Here is the initial question:
    \n ------- \n
    {question}
    \n ------- \n
    Formulate an improved question: `
  );

  // Prompt
  const chain = prompt.pipe(model).pipe(new StringOutputParser());
  const betterQuestion = await chain.invoke({ question: state.question });

  return {
    question: betterQuestion,
  };
}

/**
 * Determine whether to generate an answer, or re-generate a question.
 *
 * @param {typeof GraphState.State} state The current state of the graph.
 * @returns {"transformQuery" | "generate"} Next node to cacll
 */
function decideToGenerate(state: typeof GraphState.State) {
  console.log("---DECIDE TO GENERATE---");

  const filteredDocs = state.documents;
  if (filteredDocs.length === 0) {
    // All documents have been filtered checkRelevance
    // We will re-generate a new query
    console.log("---DECISION: TRANSFORM QUERY---");
    return "transformQuery";
  }

  // We have relevant documents, so generate answer
  console.log("---DECISION: GENERATE---");
  return "generate";
}

/**
 * Determine whether the generation is grounded inthe document.
 *
 * @param {typeof GraphState.State} state The current state of the graph.
 * @returns {Promise<Partial<typeof GraphState.State>>} The new state object.
 */
async function generateGenerationVDocumentsGrade(
  state: typeof GraphState.State
): Promise<Partial<typeof GraphState.State>> {
  console.log("---GENERATE GENERATION vs DOCUMENTS GRADE---");

  const llmWithTools = model.withStructuredOutput(
    z
      .object({
        binaryScore: z
          .enum(["yes", "no"])
          .describe("Relevance score 'yes' or 'no'"),
      })
      .describe(
        "Grade the relevance of the retrieved documents to the question. Either 'yes' or 'no'."
      ),
    {
      name: "grade",
    }
  );

  const prompt = ChatPromptTemplate.fromTemplate(
    `You are a grader assessing whether an answer is grounded in / supported by a set of facts.
    Here are the facts:
    \n ------ \n
    {documents}
    \n ------ \n
    Here is the answer: {generation}
    Give a binary score 'yes' or 'no' to indicate whether the answer is grounded in / supported by a set of facts.`
  );

  const chain = prompt.pipe(llmWithTools);

  const score = await chain.invoke({
    documents: formatDocumentsAsString(state.documents),
    generation: state.generation,
  });

  return {
    generationVDocumentsGrade: score.binaryScore,
  };
}

function gradeGenerationVDocuments(state: typeof GraphState.State) {
  console.log("---GRADE GENERATION vs DOCUMENTS---");

  const grade = state.generationVDocumentsGrade;
  if (grade === "yes") {
    console.log("---DECISION: SUPPORTED, MOVE TO FINAL GRADE---");
    return "supported";
  }

  console.log("DECISION: NOT SUPPORTED, GENERATE AGAIN");
  return "not supported";
}

/**
 * Determines whether the generation addresses the question.
 *
 * @param {typeof GraphState.State} state The current state of the graph.
 * @returns {Promise<Partial<typeof GraphState.State>>} The new state object.
 */
async function generateGenerationVQuestionGrade(
  state: typeof GraphState.State
): Promise<Partial<typeof GraphState.State>> {
  console.log("---GENERATE GENERATION vs QUESTION GRADE---");

  const llmWithTools = model.withStructuredOutput(
    z
      .object({
        binaryScore: z
          .enum(["yes", "no"])
          .describe("Relevance score 'yes' or 'no'"),
      })
      .describe(
        "Grade the relevance of the retrieved documents to the question. Either 'yes' or 'no'."
      ),
    {
      name: "grade",
    }
  );

  const prompt = ChatPromptTemplate.fromTemplate(
    `You are a grader assessing whether an answer is useful to resolve a question.
    Here is the answer:
    \n ------ \n
    {generation}
    \n ------ \n
    Here is the question: {question}
    Give a binay score 'yes' or 'no' to indicate whether the answer is useful to resolve a question.`
  );

  const chain = prompt.pipe(llmWithTools);

  const score = await chain.invoke({
    question: state.question,
    generation: state.generation,
  });

  return {
    generationVQuestionGrade: score.binaryScore,
  };
}

function gradeGenerationVQuestion(state: typeof GraphState.State) {
  console.log("---GRADE GENERATION vs QUESTION---");

  const grade = state.generationVQuestionGrade;
  if (grade === "yes") {
    console.log("---DECISION: USEFUL---");
    return "useful";
  }

  console.log("---DECISION: NOT USEFUL---");
  return "not useful";
}

import { END, START, StateGraph } from "@langchain/langgraph";

const workflow = new StateGraph(GraphState)
  // Define the nodes
  .addNode("retrieve", retrieve)
  .addNode("gradeDocuments", gradeDocuments)
  .addNode("generate", generate)
  .addNode(
    "generateGenerationVDocumentsGrade",
    generateGenerationVDocumentsGrade
  )
  .addNode("transformQuery", transformQuery)
  .addNode(
    "generateGenerationVQuestionGrade",
    generateGenerationVQuestionGrade
  );

// Build graph
workflow.addEdge(START, "retrieve");
workflow.addEdge("retrieve", "gradeDocuments");
workflow.addConditionalEdges("gradeDocuments", decideToGenerate, {
  transformQuery: "transformQuery",
  generate: "generate",
});
workflow.addEdge("transformQuery", "retrieve");
workflow.addEdge("generate", "generateGenerationVDocumentsGrade");
workflow.addConditionalEdges(
  "generateGenerationVDocumentsGrade",
  gradeGenerationVDocuments,
  {
    supported: "generateGenerationVQuestionGrade",
    "not supported": "generate",
  }
);
workflow.addConditionalEdges(
  "generateGenerationVQuestionGrade",
  gradeGenerationVQuestion,
  {
    useful: END,
    "not useful": "transformQuery",
  }
);

// Compile
const app = workflow.compile();

// Generate the graph image

// const graph = await app.getGraphAsync();
// const image = await graph.drawMermaidPng();
// const arrayBuffer = await image.arrayBuffer();

// import { writeFileSync } from "node:fs";

// writeFileSync("./selfRagGraphState.png", new Uint8Array(arrayBuffer));

const inputs = {
  question: "Explain how the different types of agent memory work.",
};
const config = { recursionLimit: 50 };

const prettifyOutput = (output: Record<string, any>) => {
  const key = Object.keys(output)[0];
  const value = output[key];
  console.log(`Node: '${key}'`);
  if (key === "retrieve" && "documents" in value) {
    console.log(`Retrieved ${value.documents.length} documents.`);
  } else if (key === "gradeDocuments" && "documents" in value) {
    console.log(
      `Graded documents. Found ${value.documents.length} relevant document(s).`
    );
  } else {
    console.dir(value, { depth: null });
  }
};

for await (const output of await app.stream(inputs, config)) {
  prettifyOutput(output);
  console.log("\n---ITERATION END---\n");
}
