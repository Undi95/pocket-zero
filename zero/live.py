"""Live progress of the long tools (forge, deck search, ranked), shown by run.bat ui (/live).

Each run writes results/live/<tool>-<name>.json: its phase, progress, what it is doing right now,
a results table that fills as it goes, and the latest events. Writes are throttled (one every
2 s, except important ones) and atomic, and never stop the tool.
"""
import json
import os
import time

DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'live')


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
