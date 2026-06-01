import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ['PUBLIC_DEMO_MODE'] = 'true'
os.environ['PUBLIC_DEMO_DAILY_TURN_BUDGET'] = '0'
os.environ['PUBLIC_DEMO_STRICT_CONFIG'] = 'false'

import app.ui as ui


def main() -> int:
    try:
        bundle = ui._load_demo_script_bundle(allow_llm_parse=False)
        source_metadata = bundle.get('source_metadata', {})
        assert isinstance(source_metadata, dict), 'demo bundle should include source metadata'
        assert source_metadata.get('preparsed') is True, 'demo story should load the pre-parsed public snapshot'
        assert source_metadata.get('loaded_from_preparsed_snapshot') is True
        assert bundle.get('scenes'), 'demo preparse snapshot should include scenes'
        assert bundle.get('knowledge'), 'demo preparse snapshot should include knowledge'
        print('[test_demo_preparse] result: PASS')
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f'[test_demo_preparse] result: FAIL -> {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
