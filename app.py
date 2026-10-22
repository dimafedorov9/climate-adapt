"""
=============================================================================
ПРОГРАММА ПОДГОТОВКИ КЛИМАТИЧЕСКИХ ДАННЫХ И ОЦЕНКИ КЛИМАТИЧЕСКИХ РИСКОВ
ДЛЯ РЕГИОНАЛЬНЫХ ПЛАНОВ АДАПТАЦИИ (тема 2.13, раздел 1)

Цепочка: загрузка суточных данных → проверка и верификация → очистка
(контроль качества) → восстановление пропусков → климатические показатели
→ тренды и сравнение периодов → районирование → карты → экспорт.
Связана с программой climate-zoning: использует её модули оформления,
кластеризации и загрузки границ, выгружает помесячные ряды в её формат.

Запуск: streamlit run app.py
=============================================================================
"""

import os
import io
import sys
import json
import pickle
import tempfile
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import params as P
from config import cf
from modules import (readers, verify, qc, fill, derived, indices, trends, clustering, viz,
                     charts, maps, basemap, interp, mapping, export)

st.set_page_config(page_title='Климатические риски: подготовка данных и районирование', layout='wide')

S = st.session_state
DEFAULTS = dict(raw=None, stations=None, checks=None, comp=None, gaps=None, qc_log=None, clean=None,
                flags=None, filled=None, fill_log=None, cv=None, D=None, ann=None, events=None,
                zoning=None, districts=None, water=None, dem=None, figs={}, tables={})
for k, v in DEFAULTS.items():
    if k not in S:
        S[k] = v if not isinstance(v, dict) else dict(v)

VAR_LABEL = {k: f'{k} — {v["name"]}' for k, v in readers.VAR_INFO.items()}
COMP_LABEL = {'tmin': 'Tмин', 'tmax': 'Tмакс', 'tmean': 'Tсредняя', 'rr_n': 'Осадки, ночь', 'rr_d': 'Осадки, день',
              'rr_s': 'Осадки, сутки', 'fm_n': 'Ветер, ночь', 'fm_d': 'Ветер, день', 'fm_s': 'Ветер, сутки',
              'ts00': 'Tпочвы 00 ВСВ', 'ts12': 'Tпочвы 12 ВСВ', 'fr': 'Класс пожарной опасности'}
for h in derived.HOURS:
    COMP_LABEL[f'tt{h}'] = f'T, {h} ВСВ'
    COMP_LABEL[f'td{h}'] = f'Td, {h} ВСВ'
    COMP_LABEL[f'rh{h}'] = f'RH, {h} ВСВ'


# ─── Вспомогательные функции интерфейса ─────────────────────────────────
def show_df(df, decimals=2, height=None):
    if df is None or len(df) == 0:
        st.caption('Нет данных.')
        return
    kw = dict(height=height) if height else {}
    st.dataframe(export.comma_frame(df, decimals), use_container_width=True, hide_index=True, **kw)


def save_table(name, df, decimals=2):
    """Запоминает таблицу для итогового архива и показывает кнопку скачивания."""
    S.tables[name] = (df, decimals)
    st.download_button(f'Скачать {name}.xlsx', export.xlsx_bytes({'Таблица': df}, decimals),
                       file_name=f'{name}.xlsx', key=f'dl_{name}')


def show_fig(fig, name, width=0.75):
    S.figs[name] = fig
    c = st.columns([width, 1 - width])[0] if width < 1 else st.container()
    with c:
        st.pyplot(fig, use_container_width=True)
        a, b = st.columns(2)
        a.download_button('PNG, 300 dpi', export.fig_bytes(fig, 'png'), file_name=f'{name}.png', key=f'png_{name}')
        b.download_button('PDF', export.fig_bytes(fig, 'pdf'), file_name=f'{name}.pdf', key=f'pdf_{name}')
    import matplotlib.pyplot as plt
    plt.close(fig)


def station_names():
    if S.stations is None:
        return {}
    return dict(zip(S.stations['st'], S.stations['name']))


_NAMES = {}


def refresh_names():
    global _NAMES
    _NAMES = station_names()


def st_label(s):
    return f'{_NAMES.get(s, s)} ({s})'


def current_data(prefer='filled'):
    if prefer == 'filled' and S.filled is not None:
        return S.filled
    if S.clean is not None:
        return S.clean
    return S.raw['data'] if S.raw else None


def stations_xy():
    t = S.stations
    return t.dropna(subset=['lat', 'lon']) if t is not None else None


refresh_names()
st.title('Климатические риски: подготовка данных, показатели и районирование')
st.caption('Суточные данные метеостанций → проверка → очистка → восстановление пропусков → '
           'показатели для планов адаптации → районирование → карты')

tabs = st.tabs(['1. Данные', '2. Проверка', '3. Очистка', '4. Пропуски', '5. Показатели',
                '6. Тренды и периоды', '7. Районирование', '8. Карты', '9. Экспорт'])

# ═══════════════════════════ 1. ДАННЫЕ ═══════════════════════════════════
with tabs[0]:
    st.header('Загрузка суточных данных')
    st.markdown('Файлы TT, TX, RR, FM, TS, RS, RH, FR (годовые, помесячные блоки) и справочники '
                '`index_*`. Можно загрузить ZIP-архив (в том числе с вложенными архивами) или указать '
                'папку на компьютере.')
    c1, c2 = st.columns(2)
    with c1:
        ups = st.file_uploader('ZIP-архивы с данными', type=['zip'], accept_multiple_files=True)
    with c2:
        folder = st.text_input('или путь к папке с данными', value='')
        rr_layout = st.selectbox('Раскладка значений в RR', ['auto', 'day', 'block'],
                                 format_func={'auto': 'Определить автоматически',
                                              'day': 'Ночь, день, сутки по датам',
                                              'block': 'Блоками по месяцу (первая версия файлов)'}.get)
        demo = st.checkbox('Тестовые данные (синтетические, для проверки работы программы)')
    if st.button('Прочитать данные', type='primary'):
        tmp = None
        try:
            if demo:
                tmp = tempfile.mkdtemp(prefix='ca_demo_')
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests'))
                import make_test_data
                make_test_data.main(tmp, 2014, 2017)
                src = tmp
            elif ups:
                tmp = tempfile.mkdtemp(prefix='ca_up_')
                for u in ups:
                    readers.extract_archive(u.getvalue(), os.path.join(tmp, os.path.splitext(u.name)[0]))
                src = tmp
            elif folder and os.path.isdir(folder):
                src = folder
            else:
                st.error('Загрузите архив или укажите существующую папку.')
                src = None
            if src:
                bar = st.progress(0.0, text='Чтение файлов…')
                res = readers.load_dataset(src, rr_layout, progress=lambda f, n: bar.progress(f, text=f'Чтение: {n}'))
                bar.empty()
                S.raw = res
                S.stations = readers.build_station_table(res['index_files'], res['data'])
                refresh_names()
                for k in ['checks', 'qc_log', 'clean', 'flags', 'filled', 'fill_log', 'cv', 'D', 'ann',
                          'events', 'zoning']:
                    S[k] = None
                st.success(f'Прочитано файлов: {len(res["files"])}; переменных: {len(res["data"])}; '
                           f'станций: {len(S.stations)}.')
        except Exception as e:
            st.exception(e)
    if S.raw:
        res = S.raw
        st.subheader('Реестр файлов')
        show_df(res['files'], 0, height=240)
        if res['notes']:
            with st.expander('Замечания при чтении', expanded=False):
                for n in res['notes']:
                    st.markdown(f'- {n}')
        st.subheader('Справочник станций')
        st.markdown('Координаты нужны для пространственной проверки, интерполяции и карт. Их можно '
                    'ввести в таблице или загрузить файлом (столбцы: индекс станции, широта, долгота, '
                    'высота).')
        coord = st.file_uploader('Файл с координатами (.xlsx, .csv)', type=['xlsx', 'csv'], key='coordfile')
        if coord is not None:
            try:
                from modules import data_io
                tab = data_io.load_table(coord)
                if isinstance(tab, dict):
                    tab = list(tab.values())[0]
                cmap_ = data_io.guess_station_column_map(tab)
                u = pd.DataFrame({'st': pd.to_numeric(tab[cmap_['Станция']], errors='coerce')})
                for k_from, k_to in [('Lat', 'lat'), ('Lon', 'lon'), ('Название', 'name')]:
                    if cmap_.get(k_from):
                        u[k_to] = tab[cmap_[k_from]].values
                alt_col = next((c for c in tab.columns if 'выс' in str(c).lower() or 'alt' in str(c).lower()), None)
                if alt_col:
                    u['alt'] = pd.to_numeric(tab[alt_col], errors='coerce').values
                S.stations = readers.build_station_table(res['index_files'], res['data'], u.dropna(subset=['st']))
                st.success('Координаты добавлены.')
            except Exception as e:
                st.error(f'Не удалось прочитать файл координат: {e}')
        degmin = st.checkbox('Координаты записаны в формате «градусы.минуты» — пересчитать в десятичные')
        ed = st.data_editor(S.stations, use_container_width=True, hide_index=True, key='st_editor',
                            column_config={'st': 'Индекс', 'name': 'Название', 'region': 'Регион',
                                           'lat': 'Широта', 'lon': 'Долгота', 'alt': 'Высота, м',
                                           'vars': 'Переменные'})
        if st.button('Сохранить справочник'):
            ed = ed.copy()
            if degmin:
                ed['lat'] = readers.degmin_to_decimal(ed['lat'])
                ed['lon'] = readers.degmin_to_decimal(ed['lon'])
            S.stations = ed
            refresh_names()
            st.success('Справочник сохранён.')

# ═══════════════════════════ 2. ПРОВЕРКА ═════════════════════════════════
with tabs[1]:
    st.header('Проверка и верификация исходных данных')
    if not S.raw:
        st.info('Сначала загрузите данные.')
    else:
        data = S.raw['data']
        if st.button('Выполнить проверку', type='primary') or S.checks is not None:
            if S.checks is None:
                with st.spinner('Проверка…'):
                    S.checks = verify.consistency_checks(data, S.stations)
                    S.comp = verify.completeness(data)
                    S.gaps = verify.network_gaps(data)
                    S.gross = verify.gross_error_summary(data)
            st.subheader('Согласованность форматов и значений')
            chk = S.checks.copy()
            chk['Значение'] = chk['Значение'].map(lambda v: cf(v, 2) if isinstance(v, float) else v)
            st.dataframe(chk.style.apply(lambda r: ['background-color: #FDECEA' if r['Статус'] == 'ошибка' else
                                                    ('background-color: #FFF4E0' if r['Статус'] == 'внимание' else '')
                                                    for _ in r], axis=1),
                         use_container_width=True, hide_index=True)
            save_table('table_01_проверка_согласованности', S.checks)
            st.subheader('Полнота рядов')
            var = st.selectbox('Переменная', sorted(data), format_func=VAR_LABEL.get, key='comp_var')
            fig = charts.completeness_heatmap(S.comp, var, station_names())
            show_fig(fig, f'fig_полнота_{var}', 0.9)
            summ = S.comp.groupby(['Переменная', 'Регион'])['Полнота, %'].agg(['mean', 'min']).reset_index()
            summ.columns = ['Переменная', 'Регион', 'Средняя полнота, %', 'Минимальная полнота, %']
            show_df(summ, 1)
            save_table('table_02_полнота_по_станциям_и_годам', S.comp, 1)
            st.subheader('Пропуски по всей сети одновременно')
            st.caption('Периоды, когда данных нет более чем на половине станций региона — сбои передачи, '
                       'которые нельзя восстановить по соседним станциям.')
            show_df(S.gaps, 1)
            save_table('table_03_пропуски_по_сети', S.gaps, 1)
            st.subheader('Значения вне физических пределов (до очистки)')
            show_df(S.gross, 1)

# ═══════════════════════════ 3. ОЧИСТКА ══════════════════════════════════
with tabs[2]:
    st.header('Очистка данных: контроль качества')
    if not S.raw:
        st.info('Сначала загрузите данные.')
    else:
        data = S.raw['data']
        with st.expander('Правила и пороги', expanded=False):
            st.markdown('Значения по умолчанию — правила раздела 4.4 памятки (согласованы с составителем '
                        'данных 09.10.2026).')
            c1, c2, c3 = st.columns(3)
            P.RR_MAX_COLD = c1.number_input('Осадки: брак вне лета, мм за сутки >', value=float(P.RR_MAX_COLD))
            P.RR_MAX_WARM = c1.number_input('Осадки: брак летом, мм за сутки ≥', value=float(P.RR_MAX_WARM))
            P.NETWORK_MEDIAN_DEV_T = c2.number_input('Отклонение от медианы сети, °C', value=float(P.NETWORK_MEDIAN_DEV_T))
            P.SPATIAL_Z_SUSPECT = c2.number_input('Проверка по соседям: |z| сомнительно', value=float(P.SPATIAL_Z_SUSPECT))
            P.SPATIAL_Z_REJECT = c2.number_input('Проверка по соседям: |z| брак', value=float(P.SPATIAL_Z_REJECT))
            P.NEIGHBOR_MAX_KM = c3.number_input('Радиус поиска соседей, км', value=float(P.NEIGHBOR_MAX_KM))
            P.FM_ROUND_MIN_STATIONS = int(c3.number_input('Ветер: круглые значения одновременно на ≥ станциях',
                                                          value=int(P.FM_ROUND_MIN_STATIONS), step=1))
            sp = st.text_input('Периоды повышенной строгости (начало–конец; через «;»)',
                               value='; '.join(f'{a}–{b}' for a, b in P.STRICT_PERIODS))
            try:
                P.STRICT_PERIODS = [tuple(x.strip().split('–')) for x in sp.split(';') if '–' in x]
            except Exception:
                pass
            sts = S.stations
            fm_st = st.multiselect('Ветер: станции, где завышенные значения делятся на 10 (две цифры вместо одной)',
                                   [int(x) for x in sts['st']], default=[s for s in P.FM_DIVIDE_BY_10 if s in sts['st'].values]
                                   or [int(s) for s in sts['st'] if 'барабинск' in str(station_names().get(s, '')).lower()],
                                   format_func=st_label)
            fm_year = st.number_input('… начиная с года', value=2018, step=1)
            P.FM_DIVIDE_BY_10 = {s: int(fm_year) for s in fm_st}
        c1, c2, c3, c4 = st.columns(4)
        opts = dict(isolation_forest=c1.checkbox('Isolation Forest (TX)', value=True),
                    persistence=c2.checkbox('Поиск «залипших» значений', value=True),
                    correct_sign=c3.checkbox('Исправлять ошибку знака Tмакс', value=False),
                    fm_kmh_convert=c4.checkbox('Ветер в км/ч пересчитывать (/3,6), а не браковать', value=False))
        if st.button('Выполнить контроль качества', type='primary'):
            bar = st.progress(0.0)
            S.qc_log = qc.run_qc(data, S.stations, opts, progress=lambda f, n: bar.progress(f, text=f'Проверка: {n}'))
            bar.empty()
            S.clean, S.flags = qc.apply_qc(data, S.qc_log)
            S.filled = None
        if S.qc_log is not None:
            log = S.qc_log
            summ = qc.qc_summary(log)
            if summ.empty:
                st.success('Ошибок не выявлено.')
            else:
                st.subheader('Сводка')
                show_fig(charts.qc_summary_bar(summ), 'fig_контроль_качества', 0.6)
                show_df(summ, 0)
                save_table('table_04_контроль_качества_сводка', summ, 0)
                st.subheader('Сомнительные значения')
                st.caption('Сомнительные значения по умолчанию остаются в данных. Отметьте правила, по которым '
                           'их тоже нужно удалить, и примените.')
                sus_rules = sorted(log[log['Действие'] == 'сомнительно']['Правило'].unique())
                rej = st.multiselect('Удалять сомнительные значения по правилам', sus_rules,
                                     format_func=lambda r: f'{r} — {qc.RULES.get(r, r)}')
                if st.button('Применить'):
                    S.clean, S.flags = qc.apply_qc(data, log, reject_suspect_rules=rej)
                    S.filled = None
                    st.success('Очищенные данные обновлены.')
                st.subheader('Журнал')
                c1, c2, c3 = st.columns(3)
                fv = c1.multiselect('Переменная', sorted(log['Переменная'].unique()))
                fa = c2.multiselect('Действие', sorted(log['Действие'].unique()))
                fs = c3.multiselect('Станция', sorted(int(x) for x in log['Станция'].unique()), format_func=st_label)
                L = log[log['Основное']]
                if fv:
                    L = L[L['Переменная'].isin(fv)]
                if fa:
                    L = L[L['Действие'].isin(fa)]
                if fs:
                    L = L[L['Станция'].isin(fs)]
                L = L.assign(Название=L['Станция'].map(station_names()))
                show_df(L[['Переменная', 'Компонент', 'Станция', 'Название', 'Дата', 'Значение', 'Новое значение',
                           'Действие', 'Правило', 'Описание правила', 'Комментарий']], 1, height=320)
                save_table('table_05_журнал_контроля_качества', log[log['Основное']].drop(columns='Основное'), 2)
                st.subheader('Ряд станции до и после очистки')
                c1, c2, c3 = st.columns(3)
                v = c1.selectbox('Переменная', [x for x in ['TX', 'TT', 'RR', 'FM', 'TS', 'RS'] if x in data],
                                 key='qc_v')
                comps = [c for c in qc.VAR_COMPS[v] if c in data[v].columns]
                cpt = c2.selectbox('Компонент', comps, format_func=lambda c: COMP_LABEL.get(c, c), key='qc_c')
                cand = [int(x) for x in L[(L['Переменная'] == v)]['Станция'].value_counts().index]
                cand += [int(x) for x in data[v]['st'].unique() if int(x) not in cand]
                sst = c3.selectbox('Станция', cand, format_func=st_label, key='qc_s')
                fig = charts.qc_series(data[v], cpt, sst, log, S.clean[v],
                                       title=f'{COMP_LABEL.get(cpt, cpt)}: {st_label(sst)}',
                                       ylabel=readers.VAR_INFO[v]['unit'])
                show_fig(fig, f'fig_очистка_{v}_{cpt}_{sst}', 0.85)

# ═══════════════════════════ 4. ПРОПУСКИ ═════════════════════════════════
with tabs[3]:
    st.header('Восстановление пропусков')
    if S.clean is None:
        st.info('Сначала выполните очистку (вкладка 3).')
    else:
        st.markdown('Для каждой переменной выберите метод или «по проверке» — тогда из имеющихся значений '
                    f'скрывается {cf(P.FILL_CV_SHARE * 100, 0)} %, они восстанавливаются каждым методом, и '
                    'выбирается метод с наименьшей RMSE. Пропуски длиннее '
                    f'{P.FILL_MAX_GAP_DAYS} суток и сутки без данных на всех соседних станциях не заполняются.')
        opts_m = {'auto': 'По проверке (лучший метод)', 'none': 'Не восстанавливать', **fill.METHODS}
        cols = st.columns(6)
        meth = {}
        for i, v in enumerate(['TX', 'TT', 'RS', 'TS', 'RR', 'FM']):
            if v in S.clean:
                choices = ['auto', 'none'] + (['ratio', 'regress', 'knn', 'rf', 'clim'] if v in ('RR', 'FM')
                                               else ['interp', 'regress', 'idw', 'knn', 'rf', 'clim'])
                meth[v] = cols[i].selectbox(v, choices, index=choices.index(fill.DEFAULT_METHODS.get(v, 'auto')),
                                            format_func=opts_m.get, key=f'fm_{v}')
        P.FILL_MAX_GAP_DAYS = int(st.number_input('Наибольшая длина восстанавливаемого пропуска, сутки',
                                                  value=int(P.FILL_MAX_GAP_DAYS), step=5))
        if st.button('Восстановить пропуски', type='primary'):
            bar = st.progress(0.0)
            S.filled, S.fill_log, S.cv = fill.fill_dataset(S.clean, S.stations, meth,
                                                           progress=lambda f, n: bar.progress(f, text=n))
            bar.empty()
        if S.filled is not None:
            st.subheader('Результат')
            fs = fill.fill_summary(S.fill_log, S.clean)
            show_df(fs, 0)
            save_table('table_06_восстановление_сводка', fs, 0)
            if S.cv is not None and not S.cv.empty:
                st.subheader('Проверка точности методов')
                show_df(S.cv.drop(columns=['Ошибка'], errors='ignore'), 2)
                save_table('table_07_проверка_методов_восстановления', S.cv, 3)
                cc = st.selectbox('Компонент', S.cv['Компонент'].unique(), format_func=lambda c: COMP_LABEL.get(c, c))
                show_fig(charts.cv_bars(S.cv, cc), f'fig_методы_{cc}', 0.7)
            st.subheader('Восстановленные значения на графике')
            c1, c2, c3 = st.columns(3)
            v = c1.selectbox('Переменная', [x for x in ['TX', 'TT', 'RR', 'FM', 'TS', 'RS'] if x in S.filled], key='f_v')
            cpt = c2.selectbox('Компонент', [c for c in fill.FILL_COMPS[v] if c in S.filled[v].columns],
                               format_func=lambda c: COMP_LABEL.get(c, c), key='f_c')
            fl = S.fill_log[(S.fill_log['var'] == v) & (S.fill_log['comp'] == cpt)]
            cand = [int(x) for x in fl['st'].value_counts().index] or [int(x) for x in S.filled[v]['st'].unique()]
            sst = c3.selectbox('Станция', cand, format_func=st_label, key='f_s')
            fig = charts.qc_series(S.clean[v], cpt, sst, None, None, fl, title=f'{COMP_LABEL.get(cpt, cpt)}: {st_label(sst)}',
                                   ylabel=readers.VAR_INFO[v]['unit'])
            show_fig(fig, f'fig_восстановление_{v}_{cpt}_{sst}', 0.85)

# ═══════════════════════════ 5. ПОКАЗАТЕЛИ ═══════════════════════════════
with tabs[4]:
    st.header('Климатические показатели для планов адаптации')
    data = current_data()
    if data is None:
        st.info('Сначала загрузите данные.')
    else:
        src = st.radio('Данные для расчёта', ['Восстановленные', 'Очищенные (без восстановления)', 'Исходные'],
                       horizontal=True, index=0 if S.filled is not None else (1 if S.clean is not None else 2))
        use = {'Восстановленные': S.filled, 'Очищенные (без восстановления)': S.clean,
               'Исходные': S.raw['data']}[src]
        c1, c2 = st.columns(2)
        P.DRY_DAY_MM = c1.number_input('Порог сухого дня, мм (< порога)', value=float(P.DRY_DAY_MM), step=0.1)
        crit_reg = c2.selectbox('Критерии ОЯ для станций без региона', list(P.CRITERIA))
        with st.expander('Критерии ОЯ и НЯ'):
            crit = pd.DataFrame(P.CRITERIA).reset_index().rename(columns={'index': 'Критерий'})
            crit = st.data_editor(crit.astype(str), hide_index=True, use_container_width=True)
            try:
                for rg in P.CRITERIA:
                    for _, r in crit.iterrows():
                        v = r[rg]
                        P.CRITERIA[rg][r['Критерий']] = eval(v) if v.startswith('(') else float(v)
            except Exception:
                st.warning('Критерии не приняты: проверьте формат чисел.')
        if use is None:
            st.warning('Выбранный набор данных ещё не сформирован.')
        elif st.button('Рассчитать показатели', type='primary'):
            with st.spinner('Расчёт…'):
                hum = None
                if 'TT' in use and 'RS' in use:
                    terms = derived.build_terms(use['TT'], use['RS'])
                    hum = derived.daily_humidity(terms)
                D = indices.build_daily(use, hum)
                D['region'] = D['region'].fillna(crit_reg)
                S.D = D
                S.ann = indices.annual_indices(D)
                S.events = dict(heat=indices.heat_events(D), cold=indices.cold_events(D),
                                precip=indices.heavy_precip_events(D) if 'rr_n' in D else pd.DataFrame(),
                                drought=indices.drought_events(D) if 'rr_s' in D else pd.DataFrame(),
                                sukh=indices.sukhovey_events(D))
        if S.ann is not None:
            ann, D = S.ann, S.D
            yrs = (int(ann['year'].min()), int(ann['year'].max()))
            st.caption(f'Годы с достаточной полнотой: {yrs[0]}–{yrs[1]}. Год учитывается, если доступно ≥ 90 % '
                       'суточных значений нужной переменной.')
            per = st.slider('Период осреднения', yrs[0], yrs[1], yrs, key='ind_period')
            sm = indices.station_means(ann, per)
            sm = S.stations[['st', 'name']].merge(sm, on='st')
            groups = sorted({v[2] for v in indices.INDEX_INFO.values()})
            grp = st.multiselect('Группы показателей', groups, default=['Жара', 'Осадки', 'Засуха', 'Агроклимат'])
            codes = [c for c, v in indices.INDEX_INFO.items() if v[2] in grp and c in sm.columns]
            tab = sm[['st', 'name', 'region', 'Лет'] + codes].rename(columns={**{c: indices.label(c) for c in codes},
                                                                               'st': 'Станция', 'name': 'Название',
                                                                               'region': 'Регион'})
            st.subheader('Средние многолетние значения по станциям')
            show_df(tab, 1, height=330)
            save_table('table_08_показатели_по_станциям', tab, 2)
            reg = ann[ann['year'].between(*per)].groupby(['region', 'Показатель'])['Значение'].mean().unstack(0)
            reg.index = [indices.label(c) for c in reg.index]
            st.subheader('Средние по регионам')
            show_df(reg.reset_index().rename(columns={'index': 'Показатель'}), 2, height=300)
            save_table('table_09_показатели_по_регионам', reg.reset_index().rename(columns={'index': 'Показатель'}), 2)

            st.subheader('Межгодовой ход показателя')
            code = st.selectbox('Показатель', sorted(ann['Показатель'].unique()), format_func=indices.label)
            show_fig(charts.index_series(ann, code, indices.label(code)), f'fig_ряд_{code}', 0.65)
            st.caption('Пунктир — наклон Сена; * — тренд значим по тесту Манна — Кендалла (p < 0,05). Для рядов '
                       'короче 30 лет оценка предварительная.')

            st.subheader('Периоды аномально жаркой погоды и сильной жары')
            ev = S.events['heat']
            c1, c2 = st.columns(2)
            pa = c1.slider('Период 1', yrs[0], yrs[1], (yrs[0], yrs[1]), key='hp1')
            pb = c2.slider('Период 2 (необязательно)', yrs[0], yrs[1], (yrs[0], yrs[1]), key='hp2')
            recs = {}
            for rg in sorted(D['region'].dropna().unique()):
                for pp in ([pa] if pa == pb else [pa, pb]):
                    r = indices.heat_summary(ev, D, pp, rg)
                    recs[f'{rg}, {r.pop("Период")} гг.'] = r
            hs = pd.DataFrame(recs)
            hs.index.name = 'Характеристика'
            hs = hs.reset_index()
            show_df(hs, 2)
            save_table('table_10_жара_сводка', hs, 2)
            with st.expander('Базовые значения для сравнения (например, отчёт 2016 г., 1985–2014 гг.)'):
                st.caption('Введите значения базового периода; они появятся рядом с рассчитанными.')
                base = st.data_editor(pd.DataFrame({'Характеристика': hs['Характеристика'], 'Базовый период': [None] * len(hs)}),
                                      hide_index=True, use_container_width=True, key='base_heat')
                if base['Базовый период'].notna().any():
                    cmpb = hs.merge(base, on='Характеристика')
                    show_df(cmpb, 2)
            with st.expander('Перечень периодов АЖ и СЖ'):
                show_df(ev.assign(Название=ev['st'].map(station_names())), 1, height=260)
                save_table('table_11_периоды_жары', ev, 1)

            st.subheader('Бездождные периоды и периоды без снегопада')
            c1, c2 = st.columns(2)
            ps = c1.slider('Период', yrs[0], yrs[1], yrs, key='dp')
            Dp = D[D['date'].dt.year.between(ps[0] - 1, ps[1])]
            for season, kind, ttl in [('тёплый', 'dry', 'Бездождные периоды, тёплый сезон (IV–X)'),
                                      ('холодный', 'dry', 'Бездождные периоды, холодный сезон (XI–III)'),
                                      ('холодный', 'nosnow', 'Периоды без снегопада, холодный сезон (XI–III)')]:
                spl, ok = indices.dry_spells(Dp, season, kind=kind)
                ok = ok[ok['Сезон'].between(*ps)]
                spl = spl[spl['Сезон'].between(*ps)] if not spl.empty else spl
                by_st, by_reg = indices.spell_class_table(spl, ok)
                st.markdown(f'**{ttl}.** Учтено сезонов: {int(ok["Учтён"].sum())} из {len(ok)} '
                            f'(сезон с пропусками > {cf(P.SEASON_MAX_MISSING * 100, 0)} % исключается).')
                if not by_reg.empty:
                    show_df(by_reg, 2)
                    show_fig(charts.spell_classes_bar(by_reg, ttl), f'fig_{kind}_{season}', 0.6)
                    save_table(f'table_12_{kind}_{season}', by_reg, 2)
                    with st.expander('По станциям'):
                        show_df(by_st.assign(Название=by_st['st'].map(station_names())), 2)
            st.subheader('Опасные явления: перечни')
            for k, ttl in [('precip', 'Сильные осадки и снег'), ('cold', 'Сильный мороз, аномально холодная погода'),
                           ('drought', 'Атмосферная засуха'), ('sukh', 'Суховеи')]:
                e = S.events.get(k)
                with st.expander(f'{ttl}: {0 if e is None else len(e)} случаев'):
                    if e is not None and len(e):
                        show_df(e.assign(Название=e['st'].map(station_names())), 1, height=240)
                        save_table(f'table_13_события_{k}', e, 1)
            st.subheader('Экстремумы и повторяемость')
            ext = indices.extremes_table(D[D['year'].between(*per)], ann[ann['year'].between(*per)])
            ext = S.stations[['st', 'name']].merge(ext, on='st')
            show_df(ext, 1, height=260)
            save_table('table_14_экстремумы_повторяемость', ext, 1)
            st.caption('Уровни повторяемости по распределению Гумбеля и число случаев АЖ, возможное раз в 50 лет '
                       '(распределение Пуассона), при длине ряда < 30 лет носят оценочный характер.')

# ═══════════════════════════ 6. ТРЕНДЫ ═══════════════════════════════════
with tabs[5]:
    st.header('Тренды и сравнение периодов')
    if S.ann is None:
        st.info('Сначала рассчитайте показатели (вкладка 5).')
    else:
        ann = S.ann
        codes = st.multiselect('Показатели', sorted(ann['Показатель'].unique()), format_func=indices.label,
                               default=[c for c in ['Tср_год', 'TXx', 'SU30', 'АЖ_случаев', 'PRCPTOT', 'CDD', 'RX1day',
                                                    'Сумма10', 'ГТК'] if c in ann['Показатель'].unique()])
        by = st.radio('Уровень', ['region', 'st'], format_func={'region': 'Средние по региону', 'st': 'По станциям'}.get,
                      horizontal=True)
        tt = trends.trend_table(ann, codes, by=by)
        if not tt.empty:
            tt['Показатель'] = tt['Показатель'].map(indices.label)
            if by == 'st':
                tt.insert(1, 'Название', tt['Станция'].map(station_names()))
            show_df(tt, 3)
            save_table('table_15_тренды', tt, 3)
            st.caption('Наклон Сена — изменение за 10 лет; тест Манна — Кендалла с поправкой Хамеда — Рао на '
                       'автокорреляцию; тест Петтитта — год наиболее вероятной смены режима.')
        yrs = (int(ann['year'].min()), int(ann['year'].max()))
        c1, c2 = st.columns(2)
        p1 = c1.slider('Период 1', yrs[0], yrs[1], (yrs[0], (yrs[0] + yrs[1]) // 2), key='cp1')
        p2 = c2.slider('Период 2', yrs[0], yrs[1], ((yrs[0] + yrs[1]) // 2 + 1, yrs[1]), key='cp2')
        cmp = trends.compare_periods(ann, p1, p2, codes, by='region')
        if not cmp.empty:
            code0 = st.selectbox('Показатель для графика', codes, format_func=indices.label, key='cmpc')
            fig = charts.period_compare_bar(cmp, code0, indices.label(code0))
            cmp['Показатель'] = cmp['Показатель'].map(indices.label)
            show_df(cmp, 3)
            save_table('table_16_сравнение_периодов', cmp, 3)
            if fig:
                show_fig(fig, f'fig_периоды_{code0}', 0.6)

# ═══════════════════════════ 7. РАЙОНИРОВАНИЕ ════════════════════════════
with tabs[6]:
    st.header('Районирование по климатическим показателям')
    if S.ann is None:
        st.info('Сначала рассчитайте показатели (вкладка 5).')
    else:
        ann = S.ann
        yrs = (int(ann['year'].min()), int(ann['year'].max()))
        per = st.slider('Период осреднения', yrs[0], yrs[1], yrs, key='z_per')
        sm = indices.station_means(ann, per)
        base = S.stations[['st', 'name', 'region', 'lat', 'lon', 'alt']].merge(sm.drop(columns='region'), on='st')
        regs = st.multiselect('Территория', sorted(base['region'].dropna().unique()),
                              default=sorted(base['region'].dropna().unique()))
        base = base[base['region'].isin(regs)].reset_index(drop=True)
        feats_all = [c for c in indices.INDEX_INFO if c in base.columns and base[c].notna().sum() >= 3
                     and base[c].std() > 0]
        default = [c for c in ['SU30', 'АЖ_случаев', 'Сумма10', 'ГТК', 'PRCPTOT', 'CDD', 'R10mm', 'Ветер15', 'Пож4',
                               'FD'] if c in feats_all]
        feats = st.multiselect('Показатели для районирования', feats_all, default=default, format_func=indices.label)
        if len(feats) < 2:
            st.warning('Выберите минимум 2 показателя.')
        else:
            red = clustering.cluster_summary_table  # noqa
            corr = base[feats].corr()
            hi = [(a, b, corr.loc[a, b]) for i, a in enumerate(feats) for b in feats[i + 1:] if abs(corr.loc[a, b]) > 0.9]
            if hi:
                st.caption('Сильно связанные показатели (|r| > 0,9) — один из пары можно исключить: ' +
                           '; '.join(f'{indices.label(a)} — {indices.label(b)} ({cf(r, 2)})' for a, b, r in hi))
            dc, Xs, _ = clustering.prepare_matrix(base, feats)
            st.caption(f'Станций с полными значениями: {len(dc)} из {len(base)}')
            kmax = max(3, min(7, len(dc) - 1))
            sil = clustering.silhouette_by_k(Xs, range(2, kmax))
            ari = clustering.ari_by_k(Xs, dc['lat'].values, range(2, kmax)) if dc['lat'].notna().all() else None
            kbest = clustering.best_k(clustering.combined_scores(sil, ari) if ari else sil)
            c1, c2, c3 = st.columns(3)
            k = c1.number_input('Число районов (k)', 2, kmax - 1 if kmax > 2 else 2, int(kbest or 2))
            alg = c2.selectbox('Алгоритм', ['kmeans', 'hierarchical', 'gmm'],
                               format_func={'kmeans': 'k-средних', 'hierarchical': 'Иерархический (Уорд)',
                                            'gmm': 'Смесь гауссовых распределений'}.get)
            order = c3.selectbox('Нумерация районов по убыванию', feats, format_func=indices.label)
            show_fig(viz.k_selection_plot(sil, ari, chosen_k=k, title='Выбор числа районов'), 'fig_выбор_k', 0.55)
            labels, s_ = clustering.run_clustering(Xs, int(k), algorithm=alg)
            dc = dc.copy()
            dc['Кластер'] = clustering.relabel_clusters_by_feature(dc, labels, order)
            S.zoning = dict(df=dc, feats=feats, k=int(k), sil=s_, period=per)
            st.success(f'Районирование выполнено: k = {int(k)}, силуэт = {cf(s_, 3)}.')
            out = dc[['st', 'name', 'region', 'Кластер'] + feats].rename(
                columns={**{c: indices.label(c) for c in feats}, 'st': 'Станция', 'name': 'Название', 'region': 'Регион',
                         'Кластер': 'Район'})
            show_df(out, 2)
            save_table('table_17_районирование_станции', out, 2)
            summ = clustering.cluster_summary_table(dc, feats, 'Кластер')
            summ.columns = [c.replace(c.split(' (')[0], indices.label(c.split(' (')[0])) if ' (' in c else c
                            for c in summ.columns]
            show_df(summ, 2)
            save_table('table_18_характеристика_районов', summ, 2)
            Xp, pm, ev = clustering.pca_transform(Xs)
            show_fig(viz.pca_biplot(Xp, pm, dc['Кластер'].values, [indices.label(f).split(',')[0] for f in feats], ev,
                                    title='Главные компоненты: станции и показатели'), 'fig_pca', 0.7)
            Z = clustering.hierarchical_linkage(Xs)
            show_fig(viz.dendrogram_plot(Z, dc['name'].astype(str).str[:12].tolist(), title='Дендрограмма (Уорд)'),
                     'fig_дендрограмма', 0.85)

# ═══════════════════════════ 8. КАРТЫ ════════════════════════════════════
with tabs[7]:
    st.header('Карты')
    stx = stations_xy()
    if stx is None or stx.empty:
        st.info('Нужны координаты станций (вкладка 1).')
    else:
        c1, c2 = st.columns(2)
        kind = c1.selectbox('Вид карты', ['Схематическая', 'По административным районам', 'Физическая',
                                          'На растровой подложке', 'Интерактивная'])
        regs_ne = basemap.region_names()
        defreg = [r for r in ['Новосибирская область', 'Омская область'] if r in regs_ne]
        terr = c2.multiselect('Территория (граница субъекта)', regs_ne, default=defreg[:1])
        rgeom = basemap.region_geometry(terr) if terr else None
        bbox = basemap.bbox_of(rgeom, pad=0.04) if rgeom is not None else basemap.bbox_of(lat=stx['lat'], lon=stx['lon'])
        # что показывать
        what_opts = ['Только станции']
        if S.zoning is not None:
            what_opts.append('Районы-кластеры')
        if S.ann is not None:
            what_opts.append('Показатель')
        what = st.radio('Что показать', what_opts, horizontal=True, index=len(what_opts) - 1 if len(what_opts) > 1 else 0)
        sdf = stx.copy()
        if rgeom is not None:
            import shapely
            inside = shapely.contains_xy(rgeom.buffer(0.05), sdf['lon'].values, sdf['lat'].values)
            sdf = sdf[inside]
        value_col, cluster_col, vlabel, dec = None, None, '', 1
        if what == 'Районы-кластеры':
            z = S.zoning['df'][['st', 'Кластер']]
            sdf = sdf.merge(z, on='st')
            cluster_col = 'Кластер'
        elif what == 'Показатель':
            ann = S.ann
            yrs = (int(ann['year'].min()), int(ann['year'].max()))
            cc1, cc2 = st.columns(2)
            code = cc1.selectbox('Показатель', sorted(ann['Показатель'].unique()), format_func=indices.label, key='map_code')
            per = cc2.slider('Период осреднения', yrs[0], yrs[1], yrs, key='map_per')
            sm = indices.station_means(ann, per)[['st', code]]
            sdf = sdf.merge(sm, on='st').dropna(subset=[code])
            value_col, vlabel = code, indices.label(code)
            dec = 0 if indices.INDEX_INFO.get(code, ('', ''))[1] in ('дни', 'случаи', 'день года', '°C·сут') else 1
            dec = st.number_input('Знаков после запятой', 0, 3, dec)
        labels_on = st.checkbox('Подписи станций', value=True)
        title = st.text_input('Заголовок', value=(vlabel.split(',')[0] if value_col else
                                                  ('Климатическое районирование' if cluster_col else 'Метеостанции')))
        # интерполяция
        field, grid = None, None
        if value_col and kind in ('Схематическая', 'Физическая', 'По административным районам', 'На растровой подложке') \
                and len(sdf) >= 4:
            im = st.selectbox('Интерполяция между станциями', ['rk', 'idw', 'nearest'],
                              format_func={'rk': 'Регрессионный кригинг (широта, долгота, высота)', 'idw': 'IDW',
                                           'nearest': 'Ближайшая станция (Тиссен)'}.get)
            step = st.select_slider('Шаг сетки, °', [0.02, 0.05, 0.1], value=0.05)
            LON, LAT = interp.make_grid(bbox, step)
            galt, alt = None, None
            if S.dem is not None and im == 'rk' and sdf['alt'].notna().all():
                galt = basemap.sample_dem(S.dem[0], S.dem[1], LAT.ravel(), LON.ravel()).reshape(LAT.shape)
                alt = sdf['alt'].values
            if im == 'rk':
                field, info = interp.regression_kriging(sdf['lat'], sdf['lon'], sdf[value_col], LAT, LON, alt=alt, galt=galt)
                st.caption(f'Доля дисперсии, объяснённая регрессией: R² = {cf(info["r2_trend"], 2)}'
                           f'{" (с высотой по ЦМР)" if info["use_alt"] else " (без высоты: загрузите рельеф)"}.')
            elif im == 'idw':
                field = interp.idw(sdf['lat'], sdf['lon'], sdf[value_col], LAT, LON)
            else:
                field = interp.nearest(sdf['lat'], sdf['lon'], sdf[value_col], LAT, LON)
            if rgeom is not None:
                field = np.where(interp.mask_by_geometry(LAT, LON, rgeom), field, np.nan)
            grid = (LON, LAT)
            with st.expander('Проверка интерполяции (поочерёдное исключение станций)'):
                cvt = interp.loo_cv(sdf['lat'], sdf['lon'], sdf[value_col],
                                    alt=sdf['alt'].values if sdf['alt'].notna().all() else None)
                show_df(cvt, 2)
        # гидросеть
        water = None
        if kind in ('Схематическая', 'По административным районам', 'Физическая'):
            wsrc = st.radio('Реки и озёра', ['Крупные (встроенные, Natural Earth)', 'Подробные (OpenStreetMap)', 'Не показывать'],
                            horizontal=True, index=0 if kind != 'Физическая' else 0)
            if wsrc.startswith('Крупные'):
                water = dict(rivers=basemap.ne_layer('rivers', bbox), lakes=basemap.ne_layer('lakes', bbox))
            elif wsrc.startswith('Подробные'):
                osm_reg = st.text_input('Название субъекта в OSM', value=terr[0] if terr else '')
                if st.button('Загрузить реки и озёра (Overpass API)'):
                    try:
                        with st.spinner('Запрос к Overpass API (до 3 минут при первой загрузке)…'):
                            S.water = basemap.fetch_osm_water(region_name=osm_reg or None, bbox=bbox)
                        st.success(f'Рек: {len(S.water["rivers"])}, водоёмов: {len(S.water["lakes"])}.')
                    except Exception as e:
                        st.error(str(e))
                water = S.water
                if water is None:
                    st.caption('Пока используются встроенные слои.')
                    water = dict(rivers=basemap.ne_layer('rivers', bbox), lakes=basemap.ne_layer('lakes', bbox))
        # районы
        districts = None
        if kind in ('По административным районам', 'Физическая', 'Схематическая', 'На растровой подложке', 'Интерактивная'):
            need = kind in ('По административным районам',)
            with st.expander('Границы административных районов', expanded=need):
                src_d = st.radio('Источник', ['Overpass API (OpenStreetMap)', 'Свой файл GeoJSON'], horizontal=True)
                if src_d.startswith('Overpass'):
                    cc1, cc2 = st.columns(2)
                    rn = cc1.text_input('Регион в OSM', value=terr[0] if terr else 'Новосибирская область', key='d_reg')
                    al = cc2.number_input('admin_level районов', 4, 10, 6)
                    if st.button('Загрузить границы районов'):
                        try:
                            key = f'{rn}_{al}'
                            cp = os.path.join(basemap.CACHE, f'districts_{abs(hash(key))}.pkl')
                            if os.path.exists(cp):
                                polys = pickle.load(open(cp, 'rb'))
                            else:
                                with st.spinner('Запрос к Overpass API…'):
                                    polys = mapping.fetch_boundaries_overpass(rn, admin_level=int(al))
                                pickle.dump(polys, open(cp, 'wb'))
                            S.districts = (S.districts or []) + [p for p in polys
                                                                 if p['name'] not in {q['name'] for q in (S.districts or [])}]
                            st.success(f'Загружено районов: {len(polys)}.')
                        except Exception as e:
                            st.error(str(e))
                else:
                    gj = st.file_uploader('GeoJSON районов', type=['geojson', 'json'])
                    if gj is not None:
                        S.districts = mapping.read_geojson_file(gj.getvalue())
                if S.districts:
                    if st.button('Очистить загруженные районы'):
                        S.districts = None
                show_d = st.checkbox('Показывать границы районов', value=bool(S.districts))
            districts = S.districts if show_d else None
        # рельеф
        dem = None
        if kind == 'Физическая':
            with st.expander('Рельеф', expanded=True):
                rsrc = st.radio('Источник рельефа', ['Тайлы высот Terrarium (AWS, SRTM)', 'Свой файл ЦМР (.tif, .asc)',
                                                     'Без рельефа'], horizontal=True)
                if rsrc.startswith('Тайлы'):
                    zoom = st.select_slider('Детальность (уровень тайлов)', [6, 7, 8, 9], value=8)
                    if st.button('Загрузить рельеф'):
                        try:
                            bar = st.progress(0.0, text='Загрузка тайлов высот…')
                            S.dem = basemap.fetch_dem(bbox, zoom=zoom, progress=lambda f: bar.progress(f))
                            bar.empty()
                        except Exception as e:
                            st.error(f'Не удалось загрузить рельеф: {e}')
                elif rsrc.startswith('Свой'):
                    df_ = st.file_uploader('ЦМР в географических координатах', type=['tif', 'tiff', 'asc'])
                    if df_ is not None:
                        try:
                            S.dem = basemap.load_dem_file(df_.getvalue(), df_.name)
                        except Exception as e:
                            st.error(f'Ошибка чтения ЦМР: {e}')
                if not rsrc.startswith('Без') and S.dem is not None:
                    dem = S.dem
            show_cities = st.checkbox('Города', value=True)
        # построение
        polys_assigned = None
        if districts and (cluster_col or value_col):
            if cluster_col:
                polys_assigned = mapping.assign_clusters_to_polygons(
                    districts, sdf.rename(columns={'lat': 'Lat', 'lon': 'Lon', 'name': 'Название'}), cluster_col='Кластер')
            elif field is not None:
                vals = interp.zonal_mean(field, grid[1], grid[0], districts)
                polys_assigned = [{**p, 'value': v} for p, v in zip(districts, vals)]
        figsize = (10, 8)
        try:
            if kind == 'Схематическая':
                mode = st.radio('Способ', ['thiessen', 'field', 'points'], horizontal=True,
                                format_func={'thiessen': 'Полигоны Тиссена', 'field': 'Изолинии (интерполяция)',
                                             'points': 'Только точки'}.get,
                                index=1 if value_col else 0)
                fig = maps.map_schematic(sdf, rgeom, bbox, cluster_col=cluster_col, value_col=value_col,
                                         value_label=vlabel, title=title, mode=mode, field=field, grid=grid,
                                         decimals=dec, labels=labels_on, rivers=water, districts=districts)
                show_fig(fig, 'map_схематическая', 0.8)
            elif kind == 'По административным районам':
                if not polys_assigned:
                    st.info('Загрузите границы районов и выберите кластеры или показатель.')
                else:
                    show_lab = st.checkbox('Подписывать районы', value=False)
                    fig = maps.map_districts(polys_assigned, sdf, bbox, key='cluster' if cluster_col else 'value',
                                             value_label=vlabel, title=title, decimals=dec, show_labels=show_lab,
                                             region_geom=rgeom, rivers=water, station_labels=labels_on)
                    show_fig(fig, 'map_районы', 0.8)
                    tbl = pd.DataFrame([{'Район': p['name'], ('Кластер' if cluster_col else 'Значение'):
                                         p.get('cluster' if cluster_col else 'value'),
                                         'Станций внутри': p.get('n_stations_inside', '')} for p in polys_assigned])
                    show_df(tbl, 2)
                    save_table('table_19_районы_карта', tbl, 2)
            elif kind == 'Физическая':
                clusters_poly = None
                if cluster_col:
                    if polys_assigned:
                        clusters_poly = polys_assigned
                    else:
                        cells = maps.thiessen(sdf, rgeom)
                        clusters_poly = [{'geometry': c, 'cluster': k} for c, k in zip(cells, sdf['Кластер']) if c is not None]
                fig = maps.map_physical(bbox, rgeom, dem=dem[0] if dem else None, dem_extent=dem[1] if dem else None,
                                        water=water, cities=basemap.ne_layer('cities', bbox) if show_cities else None,
                                        st=sdf, st_labels=labels_on, field=field if value_col else None, grid=grid,
                                        value_label=vlabel, decimals=dec, districts=districts, title=title,
                                        clusters=clusters_poly)
                show_fig(fig, 'map_физическая', 0.8)
                if dem is None:
                    st.caption('Рельеф не загружен: нажмите «Загрузить рельеф» (нужен интернет) или загрузите файл ЦМР.')
            elif kind == 'На растровой подложке':
                prov = st.selectbox('Подложка', list(basemap.TILE_PROVIDERS))
                zoom = st.select_slider('Детальность', [5, 6, 7, 8, 9], value=7)
                alpha = st.slider('Прозрачность заливки', 0.0, 1.0, 0.45)
                if st.button('Построить'):
                    try:
                        bar = st.progress(0.0, text='Загрузка тайлов…')
                        img, ext = basemap.fetch_mosaic(bbox, basemap.TILE_PROVIDERS[prov], zoom,
                                                        progress=lambda f: bar.progress(f))
                        bar.empty()
                        polys = polys_assigned
                        if polys is None and cluster_col:
                            cells = maps.thiessen(sdf, rgeom)
                            polys = [{'geometry': c, 'cluster': k, 'name': ''} for c, k in zip(cells, sdf['Кластер'])
                                     if c is not None]
                        fig = maps.map_tiles(bbox, img, ext, basemap.TILE_ATTRIB[prov], rgeom, polys,
                                             key='cluster' if cluster_col else 'value', alpha=alpha, st=sdf,
                                             st_labels=labels_on, title=title, value_label=vlabel, decimals=dec)
                        show_fig(fig, 'map_подложка', 0.8)
                    except Exception as e:
                        st.error(f'Не удалось загрузить подложку: {e}')
            else:
                from streamlit_folium import st_folium
                m = maps.folium_map(sdf.assign(name=sdf['name']), polys_assigned,
                                    key='cluster' if cluster_col else 'value', region_geom=rgeom, value_col=value_col)
                st_folium(m, height=620, use_container_width=True)
        except Exception as e:
            st.exception(e)

# ═══════════════════════════ 9. ЭКСПОРТ ══════════════════════════════════
with tabs[8]:
    st.header('Экспорт результатов')
    if not S.raw:
        st.info('Нет данных.')
    else:
        st.subheader('Связь с программой climate-zoning')
        st.markdown('Помесячные ряды температуры (по Tсредней) и осадков (суточные суммы) с листом «Справочник '
                    'станций» — в формате, который вкладка «Данные» climate-zoning распознаёт автоматически. '
                    'Месяц учитывается при наличии ≥ 25 суток с данными.')
        src = current_data()
        st.download_button('Скачать файл для climate-zoning (.xlsx)', export.zoning_xlsx(src, S.stations),
                           file_name='для_climate_zoning_помесячно.xlsx')
        st.subheader('Суточные ряды')
        which = st.radio('Набор', ['Восстановленные', 'Очищенные', 'Исходные'], horizontal=True)
        dd = {'Восстановленные': S.filled, 'Очищенные': S.clean, 'Исходные': S.raw['data']}[which]
        if dd is not None and st.button('Сформировать файл суточных рядов'):
            sheets = export.daily_export(dd, S.flags if which != 'Исходные' else None,
                                         S.fill_log if which == 'Восстановленные' else None)
            st.download_button('Скачать суточные ряды (.xlsx)', export.xlsx_bytes(sheets, 1),
                               file_name=f'суточные_ряды_{which.lower()}.xlsx')
        st.subheader('Все таблицы и рисунки')
        st.caption(f'Сохранено таблиц: {len(S.tables)}, рисунков: {len(S.figs)} (то, что открывалось на вкладках).')
        if st.button('Собрать архив'):
            files = {}
            for n, (df, d) in S.tables.items():
                files[f'tables/{n}.xlsx'] = export.xlsx_bytes({'Таблица': df}, d)
            for n, f in S.figs.items():
                files[f'figures/{n}.png'] = export.fig_bytes(f, 'png')
                files[f'figures/{n}.pdf'] = export.fig_bytes(f, 'pdf')
            if S.qc_log is not None:
                files['journal_qc.xlsx'] = export.xlsx_bytes({'Журнал': S.qc_log}, 2)
            st.download_button('Скачать архив (.zip)', export.zip_bundle(files), file_name='результаты_climate_adapt.zip')
        st.subheader('Параметры обработки')
        prm = {k: getattr(P, k) for k in dir(P) if k.isupper()}
        st.download_button('Скачать параметры (.json)', json.dumps(prm, ensure_ascii=False, indent=1, default=str),
                           file_name='параметры_обработки.json')
