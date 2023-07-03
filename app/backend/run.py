import os
from pathlib import Path

import openai
import uvicorn
from fastapi import FastAPI, WebSocket, Request
from fastapi.templating import Jinja2Templates

from app.backend.analyzer import Analyzer

BASE_DIR = Path(__file__).resolve().parent.parent

templates = Jinja2Templates(directory=str(Path(BASE_DIR, 'templates')))
log_file = "Logs"

app = FastAPI(title='Code Analyzer')


@app.get("/")
async def get(request: Request):
    """Log file viewer

    Args:
        request (Request): Default web request.

    Returns:
        TemplateResponse: Jinja template with context data.
    """
    context = {"title": "Code Analyzer tool",
               "log_file": log_file}
    return templates.TemplateResponse("index.html",
                                      {"request": request, "context": context})


@app.websocket("/ws/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            # Wait for any message from the client
            params = await websocket.receive_json()
            github_repo = params.get('github_repo')
            openai_key = params.get('openai_key')

            is_url = True if github_repo is not None and 'http' in github_repo else False
            is_openai_key = True if openai_key is not None and '-' in openai_key else False
            if is_url and is_openai_key:

                await websocket.send_text("Details saved. Please not reload the page!")
                await websocket.send_text("Set OpenAI secret key!")
                openai.api_key = openai_key
                os.environ['OPENAI_API_KEY'] = openai_key

                await websocket.send_text("Analyzing...")
                await websocket.send_text("Fetching files from public repository...")

                files, owner, repo_name = await Analyzer.aget_files_from_dir(github_repo,
                                                                             websocket)
                await websocket.send_text("Files are fetched!")
                await websocket.send_text("Iterating over fetched files...")
                if files and owner and repo_name:
                    files = await Analyzer.aread_files(file_paths=files,
                                                       owner=owner,
                                                       repo_name=repo_name)
                    items = files.items()
                    for _file, _content in items:
                        code_analyzer = Analyzer(_file, _content, websocket)
                        await code_analyzer.aanalyze_file()
                        break

                    await websocket.send_text("All files processed!")
                else:
                    await websocket.send_text("Something happen with the API")

            else:
                await websocket.send_text("Please check provided details")

        except Exception as e:
            print('error:', e)
            await websocket.send_text(f'error:{e}')

        await websocket.send_text('Analyzing completed.')


if __name__ == "__main__":
    uvicorn.run(
        "run:app",
        host="localhost",
        port=8000,
        log_level="info",
        reload=True,
        workers=1,
    )
