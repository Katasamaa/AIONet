import os
import uuid
from flask import Flask, render_template, request, jsonify
from decision_engine.decision_tree import DecisionTree

app = Flask(__name__)

decision_tree = DecisionTree()

SESSIONS_DIR = "sessions"
os.makedirs(SESSIONS_DIR, exist_ok=True)

current_session = None

def create_session():
    global current_session
    session_id = str(uuid.uuid4())[:8]
    path = os.path.join(SESSIONS_DIR, f"session_{session_id}")
    os.makedirs(path, exist_ok=True)
    current_session = path
    return session_id

def log(data):
    if current_session:
        with open(os.path.join(current_session, "log.txt"), "a") as f:
            f.write(data + "\n")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start_session", methods=["POST"])
def start_session():
    session_id = create_session()
    log(f"NEW SESSION {session_id}")
    return jsonify({"session_id": session_id})

@app.route("/select_task_type", methods=["POST"])
def select_task_type():
    data = request.json
    task_type = data.get("task_type")

    subtasks = decision_tree.get_subtasks(task_type)
    if subtasks is None:
        return jsonify({"error": "Unknown task type"}), 400

    log(f"task_type = {task_type}")

    return jsonify({"subtasks": subtasks})

@app.route("/select_subtask", methods=["POST"])
def select_subtask():
    data = request.json
    task_type = data.get("task_type")
    subtask = data.get("subtask")

    datasets = decision_tree.get_datasets(task_type, subtask)
    if datasets is None:
        return jsonify({"error": "Unknown subtask"}), 400

    log(f"subtask = {task_type}/{subtask}")

    return jsonify({"datasets": datasets})

@app.route("/submit_custom_task", methods=["POST"])
def submit_custom_task():
    data = request.json
    task_text = data.get("task")

    log(f"user_task = {task_text}")

    return jsonify({"message": "Заглушка: LLM-фильтрация находится в разработке."})

if __name__ == "__main__":
    app.run(debug=True)
