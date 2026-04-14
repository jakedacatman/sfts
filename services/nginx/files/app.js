const form = document.getElementById("classifierForm");
 
form.addEventListener("submit", async (event) => {
    event.preventDefault();
 
    try {
        const empty = document.getElementById("resultEmpty");
        const loading = document.getElementById("resultLoading");
        const filled = document.getElementById("resultFilled");

        empty.style.display = "none";
        loading.style.display = "inline";

        const jsonData = JSON.stringify(Object.fromEntries(new FormData(form).entries()));
 
        const response = await fetch("/api/identify", {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: jsonData,
        });
 
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
 
        const responseData = await response.json();
        
        loading.style.display = "none";
        filled.style.display = "inline";

        const type = document.getElementById("resultType");
        const desc = document.getElementById("resultDesc");
        
        type.textContent = responseData.prediction;
        desc.textContent = responseData.description;
        
        //form.reset();
 
    } catch (error) {
        console.log(error);
    }
});
