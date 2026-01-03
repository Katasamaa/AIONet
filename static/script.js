let selectedTaskType = null;

document.getElementById("start-session").onclick = async () => {
    let r = await fetch("/start_session", {method: "POST"});
    let j = await r.json();
    document.getElementById("session-info").innerText =
        "Сессия: " + j.session_id;
};

async function selectTaskType(type) {
    selectedTaskType = type;

    let r = await fetch("/select_task_type", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ task_type: type })
    });

    let j = await r.json();

    if (j.error) {
        alert(j.error);
        return;
    }

    let block = document.getElementById("subtasks-block");
    block.innerHTML = "<h4>Подзадачи:</h4>";

    j.subtasks.forEach(s => {
        block.innerHTML += `<button onclick="selectSubtask('${s}')">${s}</button>`;
    });
}

async function manualType() {
    const t = document.getElementById("manual-type").value;
    selectTaskType(t);
}

async function selectSubtask(subtask) {
    let r = await fetch("/select_subtask", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            task_type: selectedTaskType,
            subtask: subtask
        })
    });

    let j = await r.json();

    let block = document.getElementById("datasets-block");
    block.innerHTML = "<h4>Доступные датасеты:</h4>";

    j.datasets.forEach(d => {
        block.innerHTML += `<div>${d}</div>`;
    });
}

async function submitCustomTask() {
    let text = document.getElementById("custom-task").value;

    let r = await fetch("/submit_custom_task", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ task: text })
    });

    let j = await r.json();
    document.getElementById("result").innerText = j.message;
}
