(function () {
  const elements = {
    sources: document.getElementById("analysisSources"),
    stats: document.getElementById("analysisStats"),
    bars: document.getElementById("analysisBars"),
    cloud: document.getElementById("analysisCloud"),
    evidence: document.getElementById("analysisEvidence"),
    status: document.getElementById("analysisStatus"),
  };

  if (!elements.sources || !elements.stats || !elements.bars || !elements.cloud || !elements.evidence || !elements.status) {
    return;
  }

  const githubUsername = "AvikshithReddy";
  const sourceWeights = {
    portfolio: 1.15,
    resume: 1.35,
    github: 1.0,
  };

  const resumeChunks = [
    {
      origin: "resume",
      title: "Resume Summary",
      text: "Data Scientist and AI/ML Engineer building production machine learning and generative AI systems for large-scale structured and unstructured data. Experienced in Python, SQL, and cloud platforms with end-to-end ML lifecycle ownership including data pipelines, experimentation, deployment, and monitoring. Develops LLM-powered applications, RAG pipelines, and scalable AI workflows that transform complex data into decision intelligence and measurable business impact.",
    },
    {
      origin: "resume",
      title: "Resume Skills",
      text: "Python, SQL, FastAPI, Pandas, NumPy, PySpark, scikit-learn, PyTorch, TensorFlow, Git, Docker, Power BI, Tableau, LLMs, Agentic AI, Multi-Agent Systems, Retrieval-Augmented Generation, Prompt Engineering, LLM Evaluation, Guardrails, NLP Pipelines, Embeddings, Semantic Search, Vector Databases, LangChain, Transformers, supervised learning, unsupervised learning, regression, classification, clustering, deep learning, time-series modeling, forecasting models, feature engineering, model evaluation, model validation, hyperparameter tuning, model interpretability, exploratory data analysis, statistical modeling, hypothesis testing, experimental design, A/B testing, uplift modeling, causal inference, Apache Spark, Databricks, Airflow, ETL pipelines, data ingestion, data wrangling, data modeling, data validation, MLflow, experiment tracking, CI/CD pipelines, model deployment, model monitoring, AWS, SageMaker, Azure ML.",
    },
    {
      origin: "resume",
      title: "Resume Experience",
      text: "Improved portfolio risk monitoring by developing supervised ML models using Python, SQL, and feature engineering pipelines on borrower datasets, applying statistical modeling, cross-validation, and evaluation metrics. Built scalable analytics pipelines using PySpark, Databricks, and distributed ETL workflows. Developed NLP and LLM-driven analytics workflows using embeddings and semantic retrieval. Operationalized the machine learning lifecycle using MLflow, Docker, Git, and CI/CD pipelines. Developed predictive machine learning models using TensorFlow and PyTorch for classification, segmentation, and forecasting. Built NLP and semantic search pipelines using HuggingFace and embedding-based retrieval.",
    },
    {
      origin: "resume",
      title: "Resume Projects",
      text: "Executive Review Intelligence Platform using RAG pipelines, vector embeddings, LLM summarization, semantic search, automated issue clustering, and review analytics. Two-Stage Recommendation System using PyTorch, LightGCN, DeepFM, ranking, retrieval, and MLflow experiment tracking. AI Teaching Assistant using OpenAI API, SentenceTransformers, hybrid TF-IDF plus FAISS retrieval, and citation-grounded answers. A/B Testing and Uplift Modeling Platform using causal inference, randomized experiments, and targeted marketing strategies. Fraud Detection MLOps System using scikit-learn, FastAPI, Docker, AWS, model monitoring, and automated pipelines.",
    },
  ];

  const fallbackGithubRepos = [
    { name: "Amazon_Executive-review-intelligence", description: "NLP and RAG", language: "Python" },
    { name: "AI_expense_Claim", description: "OCR workflow and automation", language: "Python" },
    { name: "llm-finetune-chatbot", description: "LLM fine-tuning and retrieval chatbot", language: "Python" },
    { name: "teaching-assistant-rag", description: "Course-grounded RAG assistant", language: "Python" },
    { name: "AB_test", description: "A/B testing and uplift modeling", language: "Jupyter Notebook" },
    { name: "Time_Series-Demand_Forecsating", description: "Retail demand forecasting", language: "Jupyter Notebook" },
    { name: "Consumer-Affairs---Prediction", description: "Lead conversion prediction", language: "Jupyter Notebook" },
    { name: "Movie_Recommendation-Ranking-", description: "Retrieval and ranking", language: "Python" },
    { name: "Fraud_detection", description: "Fraud detection MLOps system", language: "Python" },
    { name: "EHR", description: "Intelligent care continuity archive", language: "Jupyter Notebook" },
    { name: "Time-Series-Stock-market_-Analysis", description: "Stock market time-series analysis", language: "Jupyter Notebook" },
    { name: "tenant-matcher", description: "Matching workflow", language: "Python" },
    { name: "city-sanitation-rag", description: "Multi-source RAG assistant", language: "Python" },
  ];

  const skillDefinitions = [
    {
      label: "LLM + RAG Systems",
      aliases: [
        "llm",
        "llms",
        "large language model",
        "rag",
        "retrieval-augmented generation",
        "retrieval augmented generation",
        "semantic search",
        "embeddings",
        "faiss",
        "vector database",
        "vector databases",
        "vector store",
        "openai",
        "langchain",
        "llamaindex",
        "citation-grounded",
        "grounded answers",
      ],
    },
    {
      label: "Predictive Modeling",
      aliases: [
        "machine learning",
        "predictive",
        "prediction",
        "classification",
        "regression",
        "xgboost",
        "scikit-learn",
        "feature engineering",
        "model evaluation",
        "cross-validation",
        "fraud detection",
        "churn",
        "lead conversion",
        "risk",
      ],
    },
    {
      label: "Production ML + MLOps",
      aliases: [
        "mlflow",
        "ci/cd",
        "model deployment",
        "deployment",
        "model monitoring",
        "monitoring",
        "docker",
        "fastapi",
        "production",
        "operationalized",
        "versioning",
        "reproducible",
      ],
    },
    {
      label: "Data Pipelines",
      aliases: [
        "pipeline",
        "pipelines",
        "etl",
        "data ingestion",
        "databricks",
        "pyspark",
        "spark",
        "airflow",
        "snowflake",
        "data wrangling",
        "distributed",
      ],
    },
    {
      label: "Experimentation + Causal Inference",
      aliases: [
        "a/b testing",
        "ab testing",
        "uplift",
        "causal",
        "hypothesis testing",
        "experimental design",
        "power analysis",
        "incremental",
        "conversion lift",
        "statsmodels",
      ],
    },
    {
      label: "Deep Learning",
      aliases: [
        "deep learning",
        "pytorch",
        "tensorflow",
        "keras",
        "lstm",
        "cnn",
        "transformers",
        "neural",
        "fine-tuning",
        "fine tuned",
      ],
    },
    {
      label: "NLP + Semantic Search",
      aliases: [
        "nlp",
        "document",
        "text",
        "semantic retrieval",
        "semantic search",
        "huggingface",
        "transformer",
        "topic clustering",
        "reviews",
        "summarization",
        "classification",
      ],
    },
    {
      label: "Forecasting + Time Series",
      aliases: [
        "forecast",
        "forecasting",
        "time series",
        "time-series",
        "prophet",
        "demand",
        "stock market",
        "lstm",
      ],
    },
    {
      label: "Recommendation + Ranking",
      aliases: [
        "recommendation",
        "ranking",
        "retrieval",
        "reranking",
        "lightgcn",
        "deepfm",
        "precision@10",
        "ndcg@10",
      ],
    },
    {
      label: "Cloud ML Platforms",
      aliases: [
        "aws",
        "azure",
        "gcp",
        "sagemaker",
        "azure ml",
        "redshift",
        "lambda",
        "ec2",
      ],
    },
    {
      label: "Computer Vision + OCR",
      aliases: [
        "ocr",
        "computer vision",
        "receipt",
        "field extraction",
        "docling",
        "image",
      ],
    },
    {
      label: "Analytics + BI",
      aliases: [
        "power bi",
        "tableau",
        "dashboard",
        "dashboards",
        "visualization",
        "reporting",
      ],
    },
  ];

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function escapeRegex(value) {
    return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  function countOccurrences(text, alias) {
    const normalizedAlias = alias.toLowerCase();
    const pattern = new RegExp(`\\b${escapeRegex(normalizedAlias)}\\b`, "g");
    const matches = text.match(pattern);
    return matches ? matches.length : 0;
  }

  function dedupe(items) {
    return Array.from(new Set(items));
  }

  function formatDate(value) {
    if (!value) return "";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "";
    return date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  }

  function makeChunk(origin, title, text) {
    return {
      origin,
      title,
      text: String(text || "").toLowerCase(),
      weight: sourceWeights[origin] || 1,
    };
  }

  function normalizeTextList(values) {
    return values.filter(Boolean).map(function (value) {
      return String(value).trim();
    });
  }

  function portfolioChunksFromJson(data) {
    const chunks = [];

    if (data.summary) {
      chunks.push(makeChunk("portfolio", "Portfolio Summary", data.summary));
    }

    Object.entries(data.skills || {}).forEach(function (entry) {
      const name = entry[0];
      const items = normalizeTextList(entry[1] || []);
      if (items.length) {
        chunks.push(makeChunk("portfolio", name, items.join(", ")));
      }
    });

    (data.projects || []).forEach(function (project) {
      const text = [
        project.name,
        project.description,
        (project.technologies || []).join(", "),
        project.achievements,
        project.category,
      ].join(" ");
      chunks.push(makeChunk("portfolio", project.name || "Portfolio Project", text));
    });

    (data.experience || []).forEach(function (role) {
      const text = [
        role.title,
        role.company,
        role.description,
        (role.responsibilities || []).join(" "),
      ].join(" ");
      chunks.push(makeChunk("portfolio", role.title || "Experience", text));
    });

    (data.interests || []).forEach(function (interest) {
      chunks.push(makeChunk("portfolio", "Interests", interest));
    });

    return chunks;
  }

  function portfolioChunksFromDom() {
    const selectors = [
      ["Portfolio Summary", "#summary"],
      ["Portfolio Skills", "#skills"],
      ["Portfolio Projects", "#projects"],
      ["Portfolio Experience", "#experience"],
    ];

    return selectors
      .map(function (entry) {
        const title = entry[0];
        const selector = entry[1];
        const node = document.querySelector(selector);
        if (!node) return null;
        return makeChunk("portfolio", title, node.textContent || "");
      })
      .filter(Boolean);
  }

  async function loadPortfolioChunks() {
    try {
      const response = await fetch("chatbot-backend/data/portfolio_data.json", { cache: "no-store" });
      if (!response.ok) {
        throw new Error("portfolio json unavailable");
      }
      const data = await response.json();
      const chunks = portfolioChunksFromJson(data);
      return { chunks, mode: "json" };
    } catch (error) {
      return { chunks: portfolioChunksFromDom(), mode: "dom" };
    }
  }

  function githubChunksFromRepos(repos) {
    return repos.map(function (repo) {
      const text = [
        repo.name,
        repo.description,
        repo.language,
        Array.isArray(repo.topics) ? repo.topics.join(", ") : "",
      ].join(" ");
      return makeChunk("github", repo.name || "GitHub Repo", text);
    });
  }

  async function loadGithubChunks() {
    try {
      const response = await fetch("https://api.github.com/users/" + githubUsername + "/repos?per_page=100", {
        headers: { Accept: "application/vnd.github+json" },
      });
      if (!response.ok) {
        throw new Error("github unavailable");
      }

      const repos = await response.json();
      const filtered = repos
        .filter(function (repo) {
          return !repo.fork;
        })
        .sort(function (a, b) {
          return new Date(b.updated_at) - new Date(a.updated_at);
        });

      return {
        chunks: githubChunksFromRepos(filtered),
        repoCount: filtered.length,
        live: true,
        latestUpdate: filtered[0] ? filtered[0].updated_at : "",
      };
    } catch (error) {
      return {
        chunks: githubChunksFromRepos(fallbackGithubRepos),
        repoCount: fallbackGithubRepos.length,
        live: false,
        latestUpdate: "",
      };
    }
  }

  function sourceLabel(origin) {
    if (origin === "portfolio") return "Portfolio";
    if (origin === "resume") return "Resume";
    return "GitHub";
  }

  function analyzeSignals(chunks) {
    const results = skillDefinitions
      .map(function (definition) {
        const evidence = [];
        const matchedSources = new Set();
        let rawScore = 0;
        let totalHits = 0;

        chunks.forEach(function (chunk) {
          const aliases = [];
          let chunkHits = 0;

          definition.aliases.forEach(function (alias) {
            const hits = countOccurrences(chunk.text, alias);
            if (hits > 0) {
              aliases.push(alias);
              chunkHits += Math.min(hits, 2);
            }
          });

          if (chunkHits > 0) {
            matchedSources.add(chunk.origin);
            totalHits += chunkHits;
            rawScore += Math.min(chunkHits, 5) * chunk.weight;
            evidence.push({
              title: chunk.title,
              origin: chunk.origin,
              aliases: dedupe(aliases).slice(0, 3),
              strength: chunkHits * chunk.weight,
            });
          }
        });

        return {
          label: definition.label,
          rawScore,
          totalHits,
          sourceCount: matchedSources.size,
          evidence: evidence.sort(function (a, b) {
            return b.strength - a.strength;
          }),
        };
      })
      .filter(function (item) {
        return item.rawScore > 0;
      })
      .sort(function (a, b) {
        if (b.rawScore !== a.rawScore) return b.rawScore - a.rawScore;
        return b.sourceCount - a.sourceCount;
      });

    const maxScore = results[0] ? results[0].rawScore : 1;

    return results.map(function (item) {
      const score = Math.max(18, Math.round((item.rawScore / maxScore) * 100));
      const sourceText = item.sourceCount === 1 ? "1 source" : item.sourceCount + " sources";
      return {
        label: item.label,
        score,
        totalHits: item.totalHits,
        sourceCount: item.sourceCount,
        sourceText: sourceText,
        evidence: item.evidence.slice(0, 3),
      };
    });
  }

  function renderSources(meta) {
    const sourceItems = [
      { icon: "fa-solid fa-diagram-project", label: "Portfolio", detail: meta.portfolioMode === "json" ? "structured project data" : "page content fallback" },
      { icon: "fa-solid fa-file-lines", label: "Resume", detail: "extracted AI/ML resume signals" },
      { icon: "fa-brands fa-github", label: "GitHub", detail: meta.githubLive ? "live repo metadata" : "repo snapshot fallback" },
    ];

    elements.sources.innerHTML = sourceItems
      .map(function (item) {
        return '<div class="analysis-source-pill"><i class="' + item.icon + '"></i><span>' + escapeHtml(item.label + " - " + item.detail) + "</span></div>";
      })
      .join("");
  }

  function renderStats(results, meta) {
    const threeSourceCount = results.filter(function (item) {
      return item.sourceCount === 3;
    }).length;

    const topScore = results[0] ? results[0].score : 0;
    const stats = [
      { value: String(results.length), label: "Skill Clusters" },
      { value: String(threeSourceCount), label: "Cross-Source Signals" },
      { value: String(meta.repoCount), label: "Repos Scanned" },
    ];

    elements.stats.innerHTML = stats
      .map(function (item) {
        return (
          '<div class="analysis-stat">' +
          '<span class="analysis-stat-value">' + escapeHtml(item.value) + "</span>" +
          '<span class="analysis-stat-label">' + escapeHtml(item.label) + "</span>" +
          "</div>"
        );
      })
      .join("");

    if (topScore) {
      elements.stats.insertAdjacentHTML(
        "beforeend",
        '<div class="analysis-stat"><span class="analysis-stat-value">' +
          escapeHtml(String(topScore)) +
          '</span><span class="analysis-stat-label">Top Signal Score</span></div>'
      );
    }
  }

  function renderBars(results) {
    const topResults = results.slice(0, 8);

    elements.bars.innerHTML = topResults
      .map(function (item) {
        return (
          '<div class="analysis-bar">' +
          '<div class="analysis-bar-head">' +
          "<span>" + escapeHtml(item.label) + "</span>" +
          '<span class="analysis-bar-meta">' + escapeHtml(item.score + "/100 - " + item.sourceText) + "</span>" +
          "</div>" +
          '<div class="analysis-bar-track"><div class="analysis-bar-fill" style="width:' + item.score + '%"></div></div>' +
          "</div>"
        );
      })
      .join("");
  }

  function renderCloud(results) {
    const topResults = results.slice(0, 12);
    const maxScore = topResults[0] ? topResults[0].score : 100;

    elements.cloud.innerHTML = topResults
      .map(function (item) {
        const scale = Math.max(0.2, item.score / maxScore);
        return '<span class="analysis-cloud-item" style="--scale:' + scale.toFixed(2) + '">' + escapeHtml(item.label) + "</span>";
      })
      .join("");
  }

  function renderEvidence(results) {
    const topResults = results.slice(0, 4);

    elements.evidence.innerHTML = topResults
      .map(function (item) {
        const lines = item.evidence.map(function (entry) {
          const aliases = entry.aliases.length ? " (" + entry.aliases.join(", ") + ")" : "";
          return sourceLabel(entry.origin) + ": " + entry.title + aliases;
        });

        return (
          '<div class="analysis-evidence-item">' +
          "<strong>" + escapeHtml(item.label) + "</strong>" +
          "<p>" + escapeHtml(lines.join(" | ")) + "</p>" +
          "</div>"
        );
      })
      .join("");
  }

  function renderStatus(meta) {
    const latest = meta.githubLatestUpdate ? " Latest repo update: " + formatDate(meta.githubLatestUpdate) + "." : "";
    elements.status.classList.toggle("live", meta.githubLive);
    elements.status.textContent = meta.githubLive
      ? "Live GitHub repo metadata loaded from api.github.com." + latest
      : "GitHub API unavailable, using a baked-in repository snapshot.";
  }

  async function init() {
    const [portfolioData, githubData] = await Promise.all([loadPortfolioChunks(), loadGithubChunks()]);
    const chunks = []
      .concat(portfolioData.chunks)
      .concat(resumeChunks.map(function (chunk) {
        return makeChunk(chunk.origin, chunk.title, chunk.text);
      }))
      .concat(githubData.chunks);

    const results = analyzeSignals(chunks);

    renderSources({
      portfolioMode: portfolioData.mode,
      githubLive: githubData.live,
    });
    renderStats(results, { repoCount: githubData.repoCount });
    renderBars(results);
    renderCloud(results);
    renderEvidence(results);
    renderStatus({
      githubLive: githubData.live,
      githubLatestUpdate: githubData.latestUpdate,
    });
  }

  init();
})();
