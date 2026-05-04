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
  await page.getByRole("button", { name: "What should I do after moving to Oslo?" }).click();

  await expect(page.getByText("Start with registration, tax, and the relevant Oslo services. [S1]")).toBeVisible();
  await expect(page.getByTestId("citation-list")).toContainText("UDI");
  await expect(page.getByTestId("citation-list").getByRole("link")).toHaveCount(1);
  await expect(page.getByRole("link", { name: /Moving to Norway/ })).toHaveAttribute(
    "href",
    "https://www.udi.no/en/#moving"
  );
  await expect(page.getByRole("button", { name: "What should I do after moving to Oslo?" })).toHaveCount(0);
});

test("enter sends, shift enter keeps a new line", async ({ page }) => {
  await page.goto("/");

  const composer = page.getByPlaceholder("Ask about UDI, NAV, tax, Oslo services, SUA, or SiO");
  await composer.fill("First line");
  await composer.press("Shift+Enter");
  await composer.pressSequentially("second line");
  await expect(composer).toHaveValue("First line\nsecond line");

  await composer.press("Enter");

  await expect(page.getByText("Start with registration, tax, and the relevant Oslo services. [S1]")).toBeVisible();
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
  const composer = page.getByPlaceholder("Ask about UDI, NAV, tax, Oslo services, SUA, or SiO");

  await composer.fill("First question");
  await composer.press("Enter");
  await expect(page.getByText("Checking sources")).toBeVisible();

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
  await page.getByRole("button", { name: "NO" }).click();
  await expect(page.getByPlaceholder("Spør om UDI, NAV, skatt, Oslo-tjenester, SUA eller SiO")).toBeVisible();

  await page.getByRole("button", { name: "Hvordan får jeg skattekort?" }).click();

  await expect.poll(() => postedLanguage).toBe("no");
  await expect(page.getByText("Du kan starte med offentlige kilder. [S1]")).toBeVisible();
});

test("refusal and disclaimer states are clear", async ({ page }) => {
  await page.goto("/");

  await page.getByPlaceholder("Ask about UDI, NAV, tax, Oslo services, SUA, or SiO").fill(
    "My application was rejected. Should I appeal?"
  );
  await page.getByRole("button", { name: "Send" }).click();

  await expect(page.getByText("Could not answer safely")).toBeVisible();
  await expect(page.getByTestId("disclaimer")).toContainText("not legal advice");
});

test("refresh clears the in-memory conversation", async ({ page }) => {
  await page.goto("/");

  await page.getByRole("button", { name: "What should I do after moving to Oslo?" }).click();
  await expect(page.getByText("Start with registration, tax, and the relevant Oslo services. [S1]")).toBeVisible();

  await page.reload();

  await expect(page.getByText("Start with registration, tax, and the relevant Oslo services. [S1]")).toHaveCount(0);
  await expect(page.getByText("Ask a practical question about moving to Oslo or Norway.")).toBeVisible();
});

test("clear chat and the title reset the conversation", async ({ page }) => {
  await page.goto("/");

  await page.getByRole("button", { name: "What should I do after moving to Oslo?" }).click();
  await expect(page.getByText("Start with registration, tax, and the relevant Oslo services. [S1]")).toBeVisible();

  await page.getByRole("button", { name: "Clear chat" }).click();
  await expect(page.getByText("Start with registration, tax, and the relevant Oslo services. [S1]")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "What should I do after moving to Oslo?" })).toBeVisible();

  await page.getByRole("button", { name: "What should I do after moving to Oslo?" }).click();
  await page.getByRole("heading", { name: "Oslo Newcomer Assistant" }).click();
  await expect(page.getByText("Start with registration, tax, and the relevant Oslo services. [S1]")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Where can students find housing support?" })).toBeVisible();
});

test("conversation history scrolls without hiding header actions", async ({ page }) => {
  await page.goto("/");
  const composer = page.getByPlaceholder("Ask about UDI, NAV, tax, Oslo services, SUA, or SiO");

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
  await expect(page.getByRole("button", { name: "What should I do after moving to Oslo?" })).toBeVisible();
});

test("mobile layout keeps the chat controls reachable", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");

  await expect(page.getByRole("heading", { name: "Oslo Newcomer Assistant" })).toBeVisible();
  await expect(page.getByRole("button", { name: "What should I do after moving to Oslo?" })).toBeVisible();
  await expect(page.getByPlaceholder("Ask about UDI, NAV, tax, Oslo services, SUA, or SiO")).toBeVisible();
  await expect(page.getByRole("button", { name: "Send" })).toBeVisible();
});
