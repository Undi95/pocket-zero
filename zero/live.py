"""Live progress of the long tools (forge, deck search, ranked), shown by run.bat ui (/live).

Each run writes results/live/<tool>-<name>.json: its phase, progress, what it is doing right now,
a results table that fills as it goes, and the latest events. Writes are throttled (one every
2 s, except important ones) and atomic, and never stop the tool.
"""
import json
import os
import socket
import subprocess
import sys
import time
import webbrowser

DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'live')


def open_ui(page='/live', port=8765):
    """Shows `page` of run.bat ui in the browser. The interface is started in its own window if it
    is not running yet: it stays open (and reachable from the local network) after the tool ends.
    PZ_NO_UI=1 skips it."""
    if os.environ.get('PZ_NO_UI'):
        return
    with socket.socket() as s:
        s.settimeout(0.5)
        running = s.connect_ex(('127.0.0.1', port)) == 0
    if running:
        webbrowser.open(f'http://localhost:{port}{page}')
        return
    kw = {'creationflags': subprocess.CREATE_NEW_CONSOLE} if os.name == 'nt' else \
        {'start_new_session': True, 'stdout': subprocess.DEVNULL, 'stderr': subprocess.DEVNULL}
    try:
        subprocess.Popen([sys.executable, '-m', 'zero.ui', '--page', page, '--port', str(port)],
                         cwd=os.path.dirname(os.path.dirname(DIR)), **kw)
    except OSError:
        pass


class Live:
    def __init__(self, tool, name, **info):
        self.path = os.path.join(DIR, f'{tool}-{name}.json')
        self.d = {'tool': tool, 'name': name, 'state': 'running', 'started': time.time(), 'phase': '',
                  'progress': None, 'current': {}, 'rows': [], 'events': [], 'info': info}
        self._last = 0.0
        self.write(True)

    def set(self, force=False, **fields):
        """phase=..., progress=(done, total), current={...}: updates and writes."""
        if 'progress' in fields and fields['progress'] is not None:
            done, total = fields['progress']
            fields['progress'] = {'done': done, 'total': total}
        self.d.update(fields)
        self.write(force)

    def row(self, key, **fields):
        """Adds or updates the results-table row `key`."""
        for r in self.d['rows']:
            if r.get('key') == key:
                r.update(fields)
                break
        else:
            self.d['rows'].append({'key': key, **fields})
        self.write()

    def rows(self, rows):
        self.d['rows'] = rows
        self.write()

    def event(self, text):
        self.d['events'] = (self.d['events'] + [[time.strftime('%H:%M:%S'), text]])[-80:]
        self.write(True)

    def done(self, state='done'):
        self.d['state'] = state
        self.write(True)

    def write(self, force=False):
        now = time.time()
        if not force and now - self._last < 2:
            return
        self._last = now
        self.d['updated'] = now
        try:
            os.makedirs(DIR, exist_ok=True)
            with open(self.path + '.tmp', 'w', encoding='utf-8') as f:
                json.dump(self.d, f, ensure_ascii=False)
            os.replace(self.path + '.tmp', self.path)
        except OSError:  # the page may be reading it (Windows): the next write will do
            pass


class Off:
    """Same interface, does nothing (a tool used as a library)."""
    def __getattr__(self, _):
        return lambda *a, **k: None
