#!/usr/bin/env python3
"""할일 관리 웹앱 - Flask 샘플"""

from flask import Flask, jsonify, request

app = Flask(__name__)

# 메모리 DB (샘플)
todos = [
    {"id": 1, "title": "프로젝트 설계서 작성", "done": False},
    {"id": 2, "title": "API 엔드포인트 구현", "done": False},
    {"id": 3, "title": "단위 테스트 작성", "done": True},
]


@app.route("/api/todos", methods=["GET"])
def get_todos():
    """할일 목록 조회"""
    return jsonify({"todos": todos})


@app.route("/api/todos", methods=["POST"])
def add_todo():
    """할일 추가"""
    data = request.get_json()
    new_id = max(t["id"] for t in todos) + 1 if todos else 1
    todo = {
        "id": new_id,
        "title": data.get("title", ""),
        "done": False,
    }
    todos.append(todo)
    return jsonify(todo), 201


@app.route("/api/todos/<int:todo_id>", methods=["PUT"])
def update_todo(todo_id):
    """할일 수정"""
    data = request.get_json()
    for todo in todos:
        if todo["id"] == todo_id:
            todo["title"] = data.get("title", todo["title"])
            todo["done"] = data.get("done", todo["done"])
            return jsonify(todo)
    return jsonify({"error": "not found"}), 404


@app.route("/api/todos/<int:todo_id>", methods=["DELETE"])
def delete_todo(todo_id):
    """할일 삭제"""
    global todos
    todos = [t for t in todos if t["id"] != todo_id]
    return jsonify({"message": "deleted"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
