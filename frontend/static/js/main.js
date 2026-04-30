const form = document.getElementById("form");
const resultEl = document.getElementById("result");
const errorEl = document.getElementById("error");
const submitBtn = document.getElementById("submitBtn");

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    resultEl.hidden = true;
    errorEl.hidden = true;
    submitBtn.disabled = true;
    submitBtn.textContent = "Calculating…";

    const payload = buildPayload();

    try {
        const [predict, recs, expls] = await Promise.allSettled([
            post("/predict", payload),
            post("/recommendations", payload),
            post("/explain", payload),
        ]);

        if (predict.status === "rejected") {
            throw new Error(
                "Could not reach the API. Is the server running on port 8000?",
            );
        }

        renderResult(
            predict.value,
            recs.status === "fulfilled" ? recs.value : { recommendations: [] },
            expls.status === "fulfilled" ? expls.value : { explanations: {} },
        );
        resultEl.hidden = false;
    } catch (err) {
        errorEl.textContent = err.message;
        errorEl.hidden = false;
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = "Assess Risk";
    }
});
