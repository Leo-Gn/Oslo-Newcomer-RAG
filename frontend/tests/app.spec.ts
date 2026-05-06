import { expect, test } from "@playwright/test";

const sourceSnapshot = {
  database_configured: true,
  total_sources: 6,
  total_chunks: 42,
  sources: [
    {
      owner: "UDI",
      url: "https://www.udi.no/en/",
      language: "en",
      category: "permits",
      intended_coverage: { topic: "immigration" },
      collected_at: "2026-02-01T10:00:00Z",
      official_last_updated_at: "2026-01-20T09:00:00Z",
      chunk_count: 12
    },
    {
      owner: "Skatteetaten",
      url: "https://www.skatteetaten.no/en/",
      language: "en",
      category: "tax",
      intended_coverage: { topic: "tax" },
      collected_at: "2026-02-03T10:00:00Z",
      official_last_updated_at: "2026-01-31T09:00:00Z",
      chunk_count: 9
    }
  ]
};

test.beforeEach(async ({ page }) => {
  await page.route("**/api/sources", async (route) => {
    await route.fulfill({ json: sourceSnapshot });
  });

  await page.route("**/api/chat", async (route) => {
    const request = route.request().postDataJSON();
    const isNorwegian = request.ui_language === "no";
    const asksLegalQuestion = String(request.question).toLowerCase().includes("rejected");

    await route.fulfill({
      json: {
        answer_id: `answer-${Date.now()}`,
        answer: asksLegalQuestion
          ? "I do not have enough support in the stored official sources to answer safely."
          : isNorwegian
            ? "Du kan starte med registrering, skatt og relevante tjenester i Oslo. [S1]"
            : "Start with registration, tax, and the relevant Oslo services. [S1]",
        refused: asksLegalQuestion,
        disclaimer: asksLegalQuestion
          ? "This is general information from official sources, not legal advice."
          : null,
        citations: asksLegalQuestion
            ? []
            : [
              {
                citation_id: "S1",
                chunk_id: "chunk-1",
                source_owner: "UDI",
                source_url: "https://www.udi.no/en/",
                section_url: "https://www.udi.no/en/#moving",
                section_heading: "Moving to Norway",
                collected_at: "2026-02-01T10:00:00Z",
                official_last_updated_at: "2026-01-20T09:00:00Z"
              },
              {
                citation_id: "S2",
                chunk_id: "chunk-2",
                source_owner: "UDI",
                source_url: "https://www.udi.no/en/",
                section_url: "https://www.udi.no/en/#moving",
                section_heading: "Moving to Norway",
                collected_at: "2026-02-01T10:00:00Z",
                official_last_updated_at: "2026-01-20T09:00:00Z"
              }
            ],
        data_currency: {
          collected_at: asksLegalQuestion ? null : "2026-02-01T10:00:00Z",
          official_last_updated_at: asksLegalQuestion ? null : "2026-01-20T09:00:00Z"
        }
      }
    });
  });
});

test("example prompts send a question and show citations", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByText("Updated data: 31 Jan 2026")).toBeVisible();
  await expect(page.getByRole("button", { name: "Clear chat" })).toHaveCount(0);
  await expect(page.locator(".example-button")).toHaveCount(3);
  await page.locator(".example-button").first().click();

  await expect(page.getByText("Start with registration, tax, and the relevant Oslo services. [S1]")).toBeVisible();
  await expect(page.getByTestId("citation-list")).toContainText("UDI");
  await expect(page.getByTestId("citation-list").getByRole("link")).toHaveCount(1);
  await expect(page.getByRole("link", { name: /Moving to Norway/ })).toHaveAttribute(
    "href",
    "https://www.udi.no/en/#moving"
  );
  await expect(page.locator(".example-button")).toHaveCount(0);
});

test("example prompts are reshuffled on reload", async ({ page }) => {
  await page.addInitScript(() => {
    let calls = 0;
    Math.random = () => {
      calls += 1;
      return sessionStorage.getItem("prompt-run") === "second" ? 0.92 - calls * 0.01 : 0.04 + calls * 0.01;
    };
  });

  await page.goto("/");
  const firstSet = await page.locator(".example-button").allInnerTexts();
  await page.evaluate(() => sessionStorage.setItem("prompt-run", "second"));
  await page.reload();
  const secondSet = await page.locator(".example-button").allInnerTexts();

  expect(firstSet).toHaveLength(3);
  expect(secondSet).toHaveLength(3);
  expect(secondSet).not.toEqual(firstSet);
});

test("clearing a pending answer prevents stale messages from returning", async ({ page }) => {
  let requestCount = 0;

  await page.route("**/api/chat", async (route) => {
    requestCount += 1;
    const currentRequest = requestCount;
    await new Promise((resolve) => setTimeout(resolve, currentRequest === 1 ? 450 : 20));

    try {
      await route.fulfill({
        json: {
          answer_id: `answer-${currentRequest}`,
          answer: currentRequest === 1 ? "This stale answer should not appear. [S1]" : "Fresh answer only. [S1]",
          refused: false,
          disclaimer: null,
          citations: [
            {
              citation_id: "S1",
              chunk_id: `chunk-${currentRequest}`,
              source_owner: "UDI",
              source_url: "https://www.udi.no/en/",
              section_url: "https://www.udi.no/en/#moving",
              section_heading: "Moving to Norway",
              collected_at: "2026-02-01T10:00:00Z",
              official_last_updated_at: "2026-01-20T09:00:00Z"
            }
          ],
          data_currency: {
            collected_at: "2026-02-01T10:00:00Z",
            official_last_updated_at: "2026-01-20T09:00:00Z"
          }
        }
      });
    } catch {
      // The UI can abort a pending request when the user starts over.
    }
  });

  await page.goto("/");
  await page.locator(".example-button").first().click();
  await expect(page.getByText("Checking official sources")).toBeVisible();

  await page.getByRole("button", { name: "Clear chat" }).click();
  await expect(page.locator(".example-button")).toHaveCount(3);
  await expect(page.getByText("Checking official sources")).toHaveCount(0);
  await page.waitForTimeout(520);
  await expect(page.getByText("This stale answer should not appear. [S1]")).toHaveCount(0);

  await page.locator(".example-button").first().click();
  await expect(page.getByText("Fresh answer only. [S1]")).toBeVisible();
  await expect(page.getByText("This stale answer should not appear. [S1]")).toHaveCount(0);
});

test("messages can be selected and copied", async ({ page, context }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"], { origin: "http://127.0.0.1:5173" });
  await page.goto("/");

  const promptText = (await page.locator(".example-button").first().innerText()).trim();
  await page.locator(".example-button").first().click();
  const answer = page.getByText("Start with registration, tax, and the relevant Oslo services. [S1]");
  const composer = page.getByPlaceholder("Type your question here...");

  await expect(answer).toBeVisible();
  await answer.click();
  await expect(composer).not.toBeFocused();

  await page.locator(".answer-block .copy-button").click();
  await expect(page.locator(".answer-block .copy-button")).toContainText("Copied");
  await expect.poll(() => page.evaluate(() => navigator.clipboard.readText())).toBe(
    "Start with registration, tax, and the relevant Oslo services. [S1]"
  );

  await page.locator(".question-row .copy-button").click();
  await expect.poll(() => page.evaluate(() => navigator.clipboard.readText())).toBe(promptText);
});

test("feedback buttons submit only answer metadata", async ({ page }) => {
  const feedbackBodies: unknown[] = [];

  await page.route("**/api/feedback", async (route) => {
    feedbackBodies.push(route.request().postDataJSON());
    await route.fulfill({
      status: 201,
      json: {
        feedback_id: "feedback-1",
        created_at: "2026-05-05T12:00:00Z",
        cleared: false
      }
    });
  });

  await page.goto("/");
  await page.locator(".example-button").first().click();
  await expect(page.getByText("Start with registration, tax, and the relevant Oslo services. [S1]")).toBeVisible();

  await page.getByRole("button", { name: "Helpful", exact: true }).click();

  await expect(page.getByRole("button", { name: "Helpful", exact: true })).toHaveAttribute("aria-pressed", "true");
  await expect.poll(() => feedbackBodies).toEqual([
    {
      answer_id: expect.any(String),
      rating: 1,
      citation_chunk_ids: ["chunk-1", "chunk-2"]
    }
  ]);
  expect(JSON.stringify(feedbackBodies[0])).not.toContain("Start with registration");
  expect(JSON.stringify(feedbackBodies[0])).not.toContain("What should I do");

  await page.getByRole("button", { name: "Helpful", exact: true }).click();

  await expect(page.getByRole("button", { name: "Helpful", exact: true })).toHaveAttribute("aria-pressed", "false");
  await expect.poll(() => feedbackBodies.length).toBe(2);
  expect(feedbackBodies[1]).toEqual({
    answer_id: expect.any(String),
    rating: 0,
    citation_chunk_ids: ["chunk-1", "chunk-2"]
  });
});

test("enter sends, shift enter keeps a new line", async ({ page }) => {
  await page.goto("/");

  const composer = page.getByPlaceholder("Type your question here...");
  await composer.fill("First line");
  await composer.press("Shift+Enter");
  await composer.pressSequentially("second line");
  await expect(composer).toHaveValue("First line\nsecond line");

  await composer.press("Enter");

  await expect(page.getByText("Start with registration, tax, and the relevant Oslo services. [S1]")).toBeVisible();
});

test("composer grows for longer drafts before scrolling", async ({ page }) => {
  await page.goto("/");

  const composer = page.getByPlaceholder("Type your question here...");
  const initialHeight = await composer.evaluate((element) => element.getBoundingClientRect().height);
  await composer.fill(
    [
      "I received a letter from UDI about my case.",
      "I want to ask a careful question about which public office I should check.",
      "Can you explain what information is usually available in the official sources?",
      "Please keep it short and cite the relevant page."
    ].join("\n")
  );
  const expanded = await composer.evaluate((element) => ({
    height: element.getBoundingClientRect().height,
    overflowY: getComputedStyle(element).overflowY
  }));
  const alignment = await page.locator(".composer").evaluate((element) => {
    const composerBox = element.getBoundingClientRect();
    const buttonBox = element.querySelector(".send-button")?.getBoundingClientRect();
    if (!buttonBox) {
      return Number.POSITIVE_INFINITY;
    }
    return Math.abs(
      composerBox.top + composerBox.height / 2 - (buttonBox.top + buttonBox.height / 2)
    );
  });
  const shellRadius = await page
    .locator(".composer")
    .evaluate((element) => Number.parseFloat(getComputedStyle(element).borderTopLeftRadius));

  expect(expanded.height).toBeGreaterThan(initialHeight);
  expect(expanded.overflowY).toBe("hidden");
  expect(alignment).toBeLessThan(3);
  expect(shellRadius).toBeLessThanOrEqual(32);
});

test("composer stays editable while an answer is pending", async ({ page }) => {
  await page.route("**/api/chat", async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 350));
    await route.fulfill({
      json: {
        answer_id: "slow-answer",
        answer: "Use the official pages for the current procedure. [S1]",
        refused: false,
        disclaimer: null,
        citations: [
          {
            citation_id: "S1",
            chunk_id: "chunk-1",
            source_owner: "UDI",
            source_url: "https://www.udi.no/en/",
            section_url: "https://www.udi.no/en/#moving",
            section_heading: "Moving to Norway",
            collected_at: "2026-02-01T10:00:00Z",
            official_last_updated_at: "2026-01-20T09:00:00Z"
          }
        ],
        data_currency: {
          collected_at: "2026-02-01T10:00:00Z",
          official_last_updated_at: "2026-01-20T09:00:00Z"
        }
      }
    });
  });

  await page.goto("/");
  const composer = page.getByPlaceholder("Type your question here...");

  await composer.fill("First question");
  await composer.press("Enter");
  await expect(page.getByText("Checking official sources")).toBeVisible();
  await expect(page.locator(".example-button")).toHaveCount(0);
  await expect(page.getByText("First question")).toBeVisible();

  await composer.fill("Follow-up typed while waiting");
  await expect(composer).toHaveValue("Follow-up typed while waiting");
  await expect(page.getByRole("button", { name: "Send" })).toBeDisabled();

  await expect(page.getByText("Use the official pages for the current procedure. [S1]")).toBeVisible();
  await expect(composer).toHaveValue("Follow-up typed while waiting");
  await expect(page.getByRole("button", { name: "Send" })).toBeEnabled();
});

test("language toggle sends Norwegian requests", async ({ page }) => {
  let postedLanguage = "en";

  await page.route("**/api/chat", async (route) => {
    const request = route.request().postDataJSON();
    postedLanguage = request.ui_language;
    await route.fulfill({
      json: {
        answer_id: "answer-no",
        answer: "Du kan starte med offentlige kilder. [S1]",
        refused: false,
        disclaimer: null,
        citations: [
          {
            citation_id: "S1",
            chunk_id: "chunk-1",
            source_owner: "Skatteetaten",
            source_url: "https://www.skatteetaten.no/en/",
            section_url: "https://www.skatteetaten.no/en/person/",
            section_heading: "Tax deduction card",
            collected_at: "2026-02-03T10:00:00Z",
            official_last_updated_at: "2026-01-31T09:00:00Z"
          }
        ],
        data_currency: {
          collected_at: "2026-02-03T10:00:00Z",
          official_last_updated_at: "2026-01-31T09:00:00Z"
        }
      }
    });
  });

  await page.goto("/");
  await page.getByRole("button", { name: "NO", exact: true }).click();
  await expect(page.getByPlaceholder("Skriv spørsmålet ditt her...")).toBeVisible();
  await expect(page.getByText("Hjelper deg med norsk byråkrati via offentlige kilder.")).toBeVisible();

  await expect(page.locator(".example-button")).toHaveCount(3);
  await page.locator(".example-button").first().click();

  await expect.poll(() => postedLanguage).toBe("no");
  await expect(page.getByText("Du kan starte med offentlige kilder. [S1]")).toBeVisible();
});

test("switching language keeps earlier turn labels stable", async ({ page }) => {
  await page.route("**/api/chat", async (route) => {
    const request = route.request().postDataJSON();
    const isNorwegian = request.ui_language === "no";
    await route.fulfill({
      json: {
        answer_id: isNorwegian ? "answer-no" : "answer-en",
        answer: isNorwegian ? "Dette svaret er på norsk. [S1]" : "This answer is in English. [S1]",
        refused: false,
        disclaimer: null,
        citations: [
          {
            citation_id: "S1",
            chunk_id: "chunk-1",
            source_owner: "UDI",
            source_url: "https://www.udi.no/en/",
            section_url: "https://www.udi.no/en/#moving",
            section_heading: "Moving to Norway",
            collected_at: "2026-02-01T10:00:00Z",
            official_last_updated_at: "2026-01-20T09:00:00Z"
          }
        ],
        data_currency: {
          collected_at: "2026-02-01T10:00:00Z",
          official_last_updated_at: "2026-01-20T09:00:00Z"
        }
      }
    });
  });

  await page.goto("/");
  await page.getByPlaceholder("Type your question here...").fill("What should I check after moving?");
  await page.getByRole("button", { name: "Send" }).click();
  await expect(page.getByText("This answer is in English. [S1]")).toBeVisible();

  await page.getByRole("button", { name: "NO", exact: true }).click();
  await page.getByPlaceholder("Skriv spørsmålet ditt her...").fill("Hva med skattekort?");
  await page.getByRole("button", { name: "Send" }).click();
  await expect(page.getByText("Dette svaret er på norsk. [S1]")).toBeVisible();

  const exchanges = page.locator("article");
  await expect(exchanges.nth(0)).toContainText("You");
  await expect(exchanges.nth(0)).toContainText("Answer");
  await expect(exchanges.nth(1)).toContainText("Du");
  await expect(exchanges.nth(1)).toContainText("Svar");
});

test("follow-up questions send session history", async ({ page }) => {
  const requests: unknown[] = [];

  await page.route("**/api/chat", async (route) => {
    const request = route.request().postDataJSON();
    requests.push(request);
    await route.fulfill({
      json: {
        answer_id: `answer-${requests.length}`,
        answer: requests.length === 1 ? "Students can check SiO housing. [S1]" : "They can also check Oslo housing information. [S1]",
        refused: false,
        disclaimer: null,
        citations: [
          {
            citation_id: "S1",
            chunk_id: "chunk-1",
            source_owner: "SiO",
            source_url: "https://sio.no/en/",
            section_url: "https://bolig.sio.no/en/",
            section_heading: "Student housing",
            collected_at: "2026-02-01T10:00:00Z",
            official_last_updated_at: "2026-01-20T09:00:00Z"
          }
        ],
        data_currency: {
          collected_at: "2026-02-01T10:00:00Z",
          official_last_updated_at: "2026-01-20T09:00:00Z"
        }
      }
    });
  });

  await page.goto("/");
  await page.getByPlaceholder("Type your question here...").fill("Where can students find housing support?");
  await page.getByRole("button", { name: "Send" }).click();
  await expect(page.getByText("Students can check SiO housing. [S1]")).toBeVisible();

  await page.getByPlaceholder("Type your question here...").fill("Anywhere else?");
  await page.getByRole("button", { name: "Send" }).click();
  await expect(page.getByText("They can also check Oslo housing information. [S1]")).toBeVisible();

  expect(requests).toHaveLength(2);
  expect(requests[1]).toMatchObject({
    question: "Anywhere else?",
    session_history: [
      { role: "user", content: "Where can students find housing support?" },
      { role: "assistant", content: "Students can check SiO housing. [S1]" }
    ]
  });
});

test("theme toggle starts light and switches to dark", async ({ page }) => {
  await page.goto("/");

  await expect(page.locator(".app-shell")).toHaveClass(/theme-light/);
  await page.getByRole("button", { name: "Switch to dark mode" }).click();
  await expect(page.locator(".app-shell")).toHaveClass(/theme-dark/);
  await page.getByRole("button", { name: "Switch to light mode" }).click();
  await expect(page.locator(".app-shell")).toHaveClass(/theme-light/);
});

test("refusal and disclaimer states are clear", async ({ page }) => {
  await page.goto("/");

  await page.getByPlaceholder("Type your question here...").fill(
    "My application was rejected. Should I appeal?"
  );
  await page.getByRole("button", { name: "Send" }).click();

  await expect(page.getByText("Could not answer safely")).toBeVisible();
  await expect(page.getByTestId("disclaimer")).toContainText("not legal advice");
});

test("refresh clears the in-memory conversation", async ({ page }) => {
  await page.goto("/");

  await page.locator(".example-button").first().click();
  await expect(page.getByText("Start with registration, tax, and the relevant Oslo services. [S1]")).toBeVisible();

  await page.reload();

  await expect(page.getByText("Start with registration, tax, and the relevant Oslo services. [S1]")).toHaveCount(0);
  await expect(page.getByText("Ask about permits, public services, work, tax, housing, or international student resources.")).toBeVisible();
});

test("clear chat and the title reset the conversation", async ({ page }) => {
  await page.goto("/");

  await page.locator(".example-button").first().click();
  await expect(page.getByText("Start with registration, tax, and the relevant Oslo services. [S1]")).toBeVisible();

  await page.getByRole("button", { name: "Clear chat" }).click();
  await expect(page.getByText("Start with registration, tax, and the relevant Oslo services. [S1]")).toHaveCount(0);
  await expect(page.locator(".example-button")).toHaveCount(3);

  await page.locator(".example-button").first().click();
  await page.getByRole("heading", { name: "Oslo Newcomer RAG" }).click();
  await expect(page.getByText("Start with registration, tax, and the relevant Oslo services. [S1]")).toHaveCount(0);
  await expect(page.locator(".example-button")).toHaveCount(3);
});

test("conversation history scrolls without hiding header actions", async ({ page }) => {
  await page.goto("/");
  const composer = page.getByPlaceholder("Type your question here...");

  for (let index = 0; index < 8; index += 1) {
    await composer.fill(`Housing follow-up ${index}`);
    await composer.press("Enter");
    await expect(page.getByText(`Housing follow-up ${index}`)).toBeVisible();
  }

  const stage = page.locator(".message-stage");
  const scrollInfo = await stage.evaluate((element) => ({
    top: element.scrollTop,
    height: element.scrollHeight,
    client: element.clientHeight
  }));

  expect(scrollInfo.height).toBeGreaterThan(scrollInfo.client);
  await page.getByRole("button", { name: "Clear chat" }).click();
  await expect(page.locator(".example-button")).toHaveCount(3);
});

test("mobile layout keeps the chat controls reachable", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");

  await expect(page.getByRole("heading", { name: "Oslo Newcomer RAG" })).toBeVisible();
  await expect(page.locator(".example-button")).toHaveCount(3);
  await expect(page.getByPlaceholder("Type your question here...")).toBeVisible();
  await expect(page.getByRole("button", { name: "Send" })).toBeVisible();
});
