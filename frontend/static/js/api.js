const API_BASE = "";

function buildPayload() {
    return {
        age_years: +document.getElementById("age").value,
        gender: +document.getElementById("gender").value,
        height: +document.getElementById("height").value,
        weight: +document.getElementById("weight").value,
        smoke: document.getElementById("smoke").checked ? 1 : 0,
        alco: document.getElementById("alco").checked ? 1 : 0,
        active: document.getElementById("active").checked ? 1 : 0,
        ap_hi: +document.getElementById("ap_hi").value || null,
        ap_lo: +document.getElementById("ap_lo").value || null,
        cholesterol: +document.getElementById("cholesterol").value || null,
        gluc: +document.getElementById("gluc").value || null,
    };
}

async function post(path, body) {
    const res = await fetch(`${API_BASE}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`${path} responded with ${res.status}`);
    return res.json();
}
