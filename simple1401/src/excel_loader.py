import os
import pandas as pd

def read_battery_excel(path):
    import os
    import pandas as pd

    # 强制使用 openpyxl 引擎，避免 xlrd 依赖
    xls = pd.ExcelFile(path, engine='openpyxl')
    names = list(xls.sheet_names)
    if not names:
        raise ValueError(f'工作簿为空: {path}')

    # 优先使用环境指定；否则按“cycle/step/record”模糊匹配
    picks_env = os.environ.get('XLSX_SHEETS', '').strip()
    picks = [n.strip() for n in picks_env.split(',') if n.strip()]

    if not picks:
        low_map = {str(n).lower(): n for n in names}
        def pick_like(patterns):
            for pat in patterns:
                pat = str(pat).lower()
                for low, orig in low_map.items():
                    if pat in low and orig not in picks:
                        picks.append(orig)
                        break
        pick_like(['cycle'])
        pick_like(['step'])
        pick_like(['record'])

    # 兜底：不足 3 张则按出现顺序补齐
    if len(picks) < 3:
        for n in names:
            if n not in picks:
                picks.append(n)
            if len(picks) >= 3:
                break

    def _read(name):
        hdr_env = os.environ.get('XLSX_HEADER_ROW', '').strip()
        header_row = None
        try:
            if hdr_env:
                header_row = max(0, int(hdr_env) - 1)
        except Exception:
            header_row = None
        df = pd.read_excel(xls, sheet_name=name, engine='openpyxl', header=header_row)
        df.columns = [str(c).strip() for c in df.columns]
        return df

    s1 = _read(picks[0])
    s2 = _read(picks[1]) if len(picks) > 1 else s1.copy()
    s3 = _read(picks[2]) if len(picks) > 2 else s2.copy()
    return s1, s2, s3
