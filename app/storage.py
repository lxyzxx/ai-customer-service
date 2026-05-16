from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from app.chunker import chunk_text
from app.retriever import Chunk


class Storage:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_schema(self) -> None:
        with self.connect() as db:
            db.executescript(
                """
                create table if not exists documents (
                    id integer primary key autoincrement,
                    title text not null,
                    content text not null,
                    created_at text not null default current_timestamp
                );

                create table if not exists chunks (
                    id integer primary key autoincrement,
                    document_id integer not null references documents(id) on delete cascade,
                    content text not null,
                    position integer not null
                );

                create table if not exists messages (
                    id integer primary key autoincrement,
                    session_id text not null,
                    role text not null,
                    content text not null,
                    created_at text not null default current_timestamp
                );
                """
            )

    def add_document(self, title: str, content: str) -> dict[str, Any]:
        chunks = chunk_text(content)
        if not title.strip():
            raise ValueError("title is required")
        if not chunks:
            raise ValueError("content is required")

        with self.connect() as db:
            cursor = db.execute(
                "insert into documents (title, content) values (?, ?)",
                (title.strip(), content.strip()),
            )
            document_id = int(cursor.lastrowid)
            db.executemany(
                "insert into chunks (document_id, content, position) values (?, ?, ?)",
                [(document_id, chunk, index) for index, chunk in enumerate(chunks)],
            )
        return {"id": document_id, "title": title.strip(), "chunk_count": len(chunks)}

    def list_documents(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                """
                select d.id, d.title, d.created_at, count(c.id) as chunk_count
                from documents d
                left join chunks c on c.document_id = d.id
                group by d.id
                order by d.id desc
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def list_chunks(self) -> list[Chunk]:
        with self.connect() as db:
            rows = db.execute(
                """
                select c.id, c.document_id, d.title, c.content, c.position
                from chunks c
                join documents d on d.id = c.document_id
                order by c.id
                """
            ).fetchall()
        return [
            Chunk(
                id=row["id"],
                document_id=row["document_id"],
                title=row["title"],
                content=row["content"],
                position=row["position"],
            )
                for row in rows
        ]

    def delete_document(self, document_id: int) -> None:
        with self.connect() as db:
            db.execute("delete from chunks where document_id = ?", (document_id,))
            db.execute("delete from documents where id = ?", (document_id,))

    def add_message(self, session_id: str, role: str, content: str) -> None:
        with self.connect() as db:
            db.execute(
                "insert into messages (session_id, role, content) values (?, ?, ?)",
                (session_id, role, content),
            )

    def get_recent_messages(self, session_id: str, limit: int = 8) -> list[dict[str, str]]:
        with self.connect() as db:
            rows = db.execute(
                """
                select role, content
                from messages
                where session_id = ?
                order by id desc
                limit ?
                """,
                (session_id, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def is_empty(self) -> bool:
        with self.connect() as db:
            count = db.execute("select count(*) as total from documents").fetchone()["total"]
        return count == 0
