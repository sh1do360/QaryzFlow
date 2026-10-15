from __future__ import annotations

from dataclasses import replace
from typing import cast

import streamlit as st

from qadam_engine import (
    CASES,
    OBJECTIVES,
    STRESS_SCENARIOS,
    BorrowerCase,
    Plan,
    amortization_schedule,
    choose_plan,
    effective_income,
    free_cash,
    make_plans,
    monthly_schedule,
    objective_explanation,
    portfolio_summary,
    recommendation_reason,
    sort_plans,
)
from qadam_ml_demo import (
    DEFAULT_ROWS,
    DEFAULT_SEED,
    RANKING_MODEL_KEYS,
    RANKING_MODEL_LABELS,
    SyntheticModelDemo,
    score_plan,
    train_synthetic_model,
)
from qadam_report import build_case_report


st.set_page_config(
    page_title="QaryzFlow · Debt Restructuring",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Обучаем учебную модель на искусственных примерах…")
def cached_synthetic_model(rows: int, seed: int) -> SyntheticModelDemo:
    return train_synthetic_model(rows=rows, seed=seed)


def money(value: float | None, compact: bool = False) -> str:
    if value is None:
        return "—"
    amount = int(round(value))
    sign = "-" if amount < 0 else ""
    absolute = abs(amount)
    if compact and absolute >= 1_000_000:
        return f"{sign}{absolute / 1_000_000:.1f} млн ₸"
    if compact and absolute >= 1_000:
        return f"{sign}{absolute / 1_000:.0f} тыс. ₸"
    return sign + f"{absolute:,}".replace(",", " ") + " ₸"


def input_number(value: int | float | None, fallback: float) -> float:
    return fallback if value is None else float(value)


st.markdown(
    """
    <style>
      :root { --ink:#132321; --muted:#6d7d79; --line:#e3ebe6; --green:#087f69; --mint:#dff5ee; --surface:#fff; --canvas:#f3f7f5; }
      html, body, [class*="css"] { font-family:'Segoe UI','Aptos',sans-serif; }
      .stApp { background:radial-gradient(ellipse at 8% 0%,rgba(208,239,225,.52),transparent 34%),var(--canvas);color:var(--ink); }
      [data-testid="stHeader"] { background:rgba(243,247,245,.88);backdrop-filter:blur(12px); }
      [data-testid="stSidebar"] { background:linear-gradient(180deg,#102d29 0%,#123630 100%);border-right:0; }
      [data-testid="stSidebar"] * { color:#eef7f3; }
      [data-testid="stSidebar"] [data-testid="stExpander"] { background:#173d37 !important;border:1px solid rgba(255,255,255,.14) !important;border-radius:14px !important; }
      [data-testid="stSidebar"] [data-testid="stExpander"] summary { background:transparent !important;color:#eef7f3 !important; }
      [data-testid="stSidebar"] input,[data-testid="stSidebar"] textarea { color:#132321 !important;-webkit-text-fill-color:#132321;background:#fff; }
      [data-testid="stSidebar"] input::placeholder,[data-testid="stSidebar"] textarea::placeholder { color:#73817c !important;opacity:1; }
      [data-testid="stSidebar"] [data-baseweb="select"] div,[data-baseweb="popover"],[data-baseweb="popover"] *,[role="listbox"],[role="listbox"] *,[role="option"] { color:#132321 !important; }
      [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color:#b8cbc4;font-size:.78rem; }
      [data-testid="stSidebar"] hr { border-color:rgba(255,255,255,.14); }
      [data-testid="stSidebar"] [role="radiogroup"] { gap:4px; }
      [data-testid="stSidebar"] [role="radiogroup"] label { border:1px solid transparent;border-radius:11px;padding:7px 9px;transition:background .16s ease,border-color .16s ease; }
      [data-testid="stSidebar"] [role="radiogroup"] label:hover { background:rgba(255,255,255,.06); }
      [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background:rgba(184,241,217,.11);border-color:rgba(184,241,217,.2); }
      .block-container { max-width:1380px;padding-top:2rem;padding-bottom:3.5rem; }
      h1,h2,h3 { font-family:'Segoe UI','Aptos',sans-serif;color:var(--ink);letter-spacing:-.035em; }
      h1 { font-size:2.15rem !important;font-weight:800 !important;margin-bottom:.25rem !important; }
      h2 { font-size:1.2rem !important;font-weight:750 !important; }
      h3 { font-size:1rem !important;font-weight:700 !important; }
      p,li { color:#40524e; }
      .brand { display:flex;align-items:center;gap:11px;padding:7px 0 23px; }
      .brand-mark { width:40px;height:40px;border-radius:14px;background:linear-gradient(145deg,#c7f5df,#93e1c2);color:#103d32;display:grid;place-items:center;font-size:21px;font-weight:800;box-shadow:0 8px 20px rgba(3,18,15,.18); }
      .brand-name { font-size:18px;font-weight:800;letter-spacing:-.04em;color:#f4fbf8; }
      .brand-caption { color:#a7c4b9;font-size:10px;letter-spacing:.16em;text-transform:uppercase;margin-top:1px; }
      .eyebrow { color:#638078;font-size:.68rem;letter-spacing:.15em;text-transform:uppercase;font-weight:800; }
      .page-subtitle { color:var(--muted);font-size:.94rem;line-height:1.55;max-width:780px;margin:0 0 1.2rem; }
      .status-pill { display:inline-flex;align-items:center;gap:7px;border:1px solid #cce8db;border-radius:999px;padding:7px 11px;background:#effaf4;color:#197454;font-weight:750;font-size:.69rem;letter-spacing:.03em;white-space:nowrap; }
      .status-dot { width:7px;height:7px;border-radius:50%;background:#29a878;box-shadow:0 0 0 3px #d8f1e6; }
      .case-context { display:flex;align-items:center;justify-content:space-between;gap:18px;background:#fff;border:1px solid var(--line);border-radius:16px;padding:13px 16px;margin:.65rem 0 1rem;box-shadow:0 5px 18px rgba(19,48,40,.035); }
      .case-context-main { font-size:.94rem;font-weight:750;color:var(--ink);margin-top:4px; }
      .case-id { color:#71847c;font-size:.66rem;letter-spacing:.13em;text-transform:uppercase;font-weight:750; }
      .panel-title { font-weight:780;color:var(--ink);font-size:1rem;margin-bottom:4px; }
      .panel-caption { font-size:.78rem;color:var(--muted);line-height:1.5;margin-bottom:14px; }
      [data-testid="stVerticalBlockBorderWrapper"] { background:#fff;border:1px solid var(--line) !important;border-radius:18px !important;box-shadow:0 8px 24px rgba(19,48,40,.045); }
      .recommend-card { background:linear-gradient(135deg,#103b33 0%,#0a7862 100%);border:1px solid rgba(190,242,218,.16);border-radius:20px;padding:24px 25px;color:#f4fff9;position:relative;overflow:hidden;box-shadow:0 16px 34px rgba(10,76,61,.17); }
      .recommend-card .eyebrow { color:#a7d9c7; }
      .recommend-card h3 { color:#fff;font-size:1.2rem !important;margin:.4rem 0 .3rem; }
      .recommend-card p { color:#d4eee4;font-size:.82rem;margin:0; }
      .recommend-number { font-weight:800;font-size:1.65rem;letter-spacing:-.05em;color:#fff;line-height:1.1; }
      .recommend-small { font-size:.72rem;color:#bbe2d3; }
      .signal { display:flex;align-items:flex-start;gap:9px;color:#40524e;font-size:.81rem;margin:10px 0; }
      .signal-mark { flex:0 0 19px;width:19px;height:19px;border-radius:50%;background:#e5f4ec;color:#087f69;display:grid;place-items:center;font-size:.7rem;font-weight:800; }
      .callout { border-radius:13px;background:#eef7f3;border:1px solid #dcece4;padding:13px 15px;color:#405a51;font-size:.8rem;line-height:1.55; }
      .callout strong { color:#164d3f; }
      .tiny-note { color:#81908b;font-size:.72rem;line-height:1.5; }
      .sidebar-section { color:#8eaea1;font-size:.65rem;font-weight:800;letter-spacing:.13em;text-transform:uppercase;margin:10px 0 5px; }
      .stButton > button,[data-testid="stDownloadButton"] > button { border-radius:11px;border:1px solid #d9e6e0;font-weight:750;transition:transform .15s ease,box-shadow .15s ease; }
      .stButton > button:hover,[data-testid="stDownloadButton"] > button:hover { border-color:#9fcab8;box-shadow:0 5px 14px rgba(20,84,67,.1);transform:translateY(-1px); }
      [data-testid="stMetric"] { background:#fff;border:1px solid var(--line);border-radius:16px;padding:16px 17px;box-shadow:0 6px 20px rgba(19,48,40,.035); }
      [data-testid="stMetricLabel"] { color:#70817c;font-size:.73rem;font-weight:700; }
      [data-testid="stMetricValue"] { color:var(--ink);font-weight:800;letter-spacing:-.045em; }
      [data-testid="stDataFrame"] { border:1px solid var(--line);border-radius:14px;overflow:hidden;background:#fff; }
      @media(max-width:800px) { .block-container { padding:1rem 1rem 2rem; } h1 {font-size:1.65rem !important;} .case-context {align-items:flex-start;} .recommend-card {padding:19px;} }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_header(title: str, subtitle: str) -> None:
    left, right = st.columns([0.76, 0.24], vertical_alignment="center")
    with left:
        st.markdown('<div class="eyebrow">QARYZFLOW · УСТОЙЧИВОЕ УРЕГУЛИРОВАНИЕ</div>', unsafe_allow_html=True)
        st.title(title)
        st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)
    with right:
        st.markdown(
            '<div style="text-align:right"><span class="status-pill"><span class="status-dot"></span>ДЕМО · ВЫМЫШЛЕННЫЕ КЕЙСЫ</span></div>',
            unsafe_allow_html=True,
        )


def render_case_page(
    case: BorrowerCase,
    objective: str,
    budget_share: float,
    max_extension_months: int,
    max_grace_months: int,
) -> None:
    plans = make_plans(
        case,
        budget_share=budget_share,
        max_extension_months=max_extension_months,
        max_grace_months=max_grace_months,
    )
    current = next(plan for plan in plans if plan.key == "current")
    recommended = choose_plan(plans, objective)
    feasible_count = sum(plan.is_feasible for plan in plans)
    frontier = [plan for plan in plans if plan.is_pareto_optimal]
    if recommended is not None:
        closest: Plan = recommended
    else:
        alternatives = [plan for plan in plans if plan.key != "current"]
        closest = min(
            alternatives,
            key=lambda plan: (
                plan.feasibility_gap,
                plan.peak_monthly_payment,
                plan.total_repayment,
            ),
            default=current,
        )
    reserve_months = case.savings / max(case.essential_expenses, 1.0)
    cash = free_cash(case)

    render_header(
        "Подбор графика урегулирования",
        "Система перебирает варианты срока и паузы, отсекает графики выше бюджетного лимита и показывает компромиссы.",
    )
    st.markdown(
        f"""
        <div class="case-context">
          <div>
            <div class="case-id">УЧЕБНЫЙ КЕЙС · {case.case_id}</div>
            <div class="case-context-main">{case.region} <span style="color:#9aa8a2">·</span> {case.segment}</div>
          </div>
          <div style="text-align:right">
            <div class="case-id">Текущая оценка бюджета</div>
            <div class="case-context-main">{money(effective_income(case))} доход после снижения</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Доход после снижения", money(effective_income(case), compact=True), f"Обычный {money(case.income, compact=True)}")
    k2.metric("Свободный бюджет", money(cash, compact=True), "До платежа по этому долгу")
    k3.metric("Подходящие графики", f"{feasible_count:,}".replace(",", " "), f"из {len(plans)} перебранных")
    k4.metric(
        "Стресс-проверки",
        f"{recommended.stress_passes}/{recommended.stress_total}" if recommended else "Нет варианта",
        "для выбранного графика" if recommended else "лимит не проходит",
    )

    st.write("")
    left, right = st.columns([1.08, 0.92], gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Свободный бюджет после максимального платежа</div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-caption">Для текущего графика учтён возможный финальный остаток; для нового — регулярный платёж после паузы.</div>', unsafe_allow_html=True)
            cash_rows = [
                {"График": "Текущий", "Свободно после платежа": cash - current.peak_monthly_payment},
                {
                    "График": "Выбранный" if recommended else "Ближайший, вне лимита",
                    "Свободно после платежа": cash - closest.peak_monthly_payment,
                },
            ]
            st.bar_chart(cash_rows, x="График", y="Свободно после платежа", color="#13866d", horizontal=True)
            st.markdown(
                f'<div class="tiny-note">Лимит платежа: <b>{money(max(0.0, cash) * budget_share)}</b> · резерв: <b>{reserve_months:.1f} мес.</b></div>',
                unsafe_allow_html=True,
            )

        st.write("")
        with st.container(border=True):
            chart_title = "Компромисс: платёж и сумма выплат"
            st.markdown(f'<div class="panel-title">{chart_title}</div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-caption">Показаны все сценарии, проходящие лимит, и текущий график для сравнения. Легенда выделяет выбранный вариант и границу Парето.</div>', unsafe_allow_html=True)
            current_chart_status = ["Текущий график"]
            if current.is_pareto_optimal:
                current_chart_status.append("граница Парето")
            if recommended and recommended.key == "current":
                current_chart_status.append("выбранный вариант")
            scatter_rows = [
                {
                    "График": "Текущий",
                    "Тип": " · ".join(current_chart_status),
                    "Максимальный платёж за месяц": current.peak_monthly_payment,
                    "Сумма выплат": current.total_repayment,
                }
            ]
            for plan in plans:
                if not plan.is_feasible or plan.key == "current":
                    continue
                is_selected = recommended is not None and plan.key == recommended.key
                if is_selected and plan.is_pareto_optimal:
                    status = "Выбранный · граница Парето"
                elif is_selected:
                    status = "Выбранный вариант"
                elif plan.is_pareto_optimal:
                    status = "Граница Парето"
                else:
                    status = "Другой допустимый график"
                scatter_rows.append(
                    {
                        "График": f"{plan.term_months} мес. / пауза {plan.grace_months}",
                        "Тип": status,
                        "Максимальный платёж за месяц": plan.peak_monthly_payment,
                        "Сумма выплат": plan.total_repayment,
                    }
                )
            st.scatter_chart(scatter_rows, x="Максимальный платёж за месяц", y="Сумма выплат", color="Тип")

    with right:
        if recommended:
            st.markdown(
                f"""
                <div class="recommend-card">
                  <div class="eyebrow">ВАРИАНТ ПО ВЫБРАННОЙ ЦЕЛИ · {OBJECTIVES[objective].upper()}</div>
                  <h3>{recommended.title}</h3>
                  <p>{recommended.description}</p>
                  <div style="height:18px"></div>
                  <div style="display:flex;gap:32px;flex-wrap:wrap">
                    <div><div class="recommend-number">{money(recommended.monthly_payment, compact=True)}</div><div class="recommend-small">обычный платёж / мес.</div></div>
                    <div><div class="recommend-number">{money(recommended.peak_monthly_payment, compact=True)}</div><div class="recommend-small">максимальный с финальным остатком</div></div>
                    <div><div class="recommend-number">{money(recommended.total_repayment, compact=True)}</div><div class="recommend-small">всего выплат за срок</div></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("По введённому бюджету подходящего графика нет. Ниже показан вариант с наименьшим превышением лимита для обсуждения, а не рекомендация.")
            st.markdown(
                f"""
                <div class="recommend-card">
                  <div class="eyebrow">НЕ ПРОХОДИТ БЮДЖЕТНЫЙ ЛИМИТ</div>
                  <h3>{closest.title}</h3>
                <p>Максимальный платёж выше установленного лимита на {money(closest.feasibility_gap)} в месяц.</p>
                  <div style="height:18px"></div>
                  <div class="recommend-number">{money(closest.monthly_payment, compact=True)}</div>
                  <div class="recommend-small">платёж после паузы · только для анализа</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")
        with st.container(border=True):
            st.markdown('<div class="panel-title">Почему выбран этот график</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="panel-caption">{objective_explanation(objective)}</div>', unsafe_allow_html=True)
            if recommended:
                for reason in recommendation_reason(case, recommended, budget_share):
                    st.markdown(f'<div class="signal"><span class="signal-mark">✓</span><span>{reason}</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="signal"><span class="signal-mark">!</span><span>Максимальный платёж ближайшего варианта превышает лимит на {money(closest.feasibility_gap)} в месяц.</span></div>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                '<div class="callout"><strong>Стресс-проверки — не прогноз.</strong> Это четыре заданных изменения дохода и расходов. Они показывают, сохраняется ли выбранный бюджетный лимит при таких условиях.</div>',
                unsafe_allow_html=True,
            )

    st.write("")
    with st.container(border=True):
        st.markdown('<div class="panel-title">Помесячный график · первые 12 месяцев</div>', unsafe_allow_html=True)
        selected_schedule_label = "Выбранный график" if recommended else "Ближайший вариант · вне лимита"
        st.markdown(
            '<div class="panel-caption">Текущий график учитывает возможный остаток долга в конце срока. Для нового графика проценты на время паузы капитализируются.</div>',
            unsafe_allow_html=True,
        )
        months = list(range(1, 13))
        st.line_chart(
            {
                "Месяц": months,
                "Текущий график": monthly_schedule(case, current, 12),
                selected_schedule_label: monthly_schedule(case, closest, 12),
            },
            x="Месяц",
            y=["Текущий график", selected_schedule_label],
            color=["#9aa9a3", "#13866d"],
        )

    with st.expander("Подробная разбивка выплат", expanded=False):
        st.caption(
            "Показывает, сколько процентов начисляется и уплачивается, какая часть платежа гасит долг, "
            "и сколько процентов добавляется к долгу во время паузы или при недостаточном платеже."
        )
        schedule_plans = [("Текущий график", current)]
        if closest.key != current.key:
            schedule_plans.append((selected_schedule_label, closest))

        for schedule_label, schedule_plan in schedule_plans:
            rows = amortization_schedule(case, schedule_plan)
            st.markdown(f"**{schedule_label}**")
            st.caption(
                f"Всего выплат: {money(sum(row.payment for row in rows))} · "
                f"начислено процентов: {money(sum(row.accrued_interest for row in rows))} · "
                f"капитализировано: {money(sum(row.capitalized_interest for row in rows))}"
            )
            st.dataframe(
                [
                    {
                        "Месяц": row.month,
                        "Долг на начало": money(row.opening_balance),
                        "Платёж": money(row.payment),
                        "Начислено процентов": money(row.accrued_interest),
                        "Уплачено процентов": money(row.interest_paid),
                        "Погашено долга": money(row.principal_repaid),
                        "Капитализировано процентов": money(row.capitalized_interest),
                        "Долг на конец": money(row.closing_balance),
                    }
                    for row in rows
                ],
                width="stretch",
                hide_index=True,
            )

    st.write("")
    with st.container(border=True):
        st.markdown('<div class="panel-title">Графики на границе компромиссов</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="panel-caption">Цель подбора: {OBJECTIVES[objective].lower()}. Показаны базовый график и недоминируемые варианты; полный перебор доступен в JSON.</div>',
            unsafe_allow_html=True,
        )
        display_plans = {plan.key: plan for plan in frontier}
        display_plans[current.key] = current
        if not recommended:
            display_plans[closest.key] = closest
        table_rows = []
        for plan in sort_plans(list(display_plans.values()), objective):
            table_rows.append(
                {
                    "Статус": (
                        "Текущий график"
                        if plan.key == "current"
                        else "Подходит по бюджету"
                        if plan.is_feasible
                        else "Выше бюджетного лимита"
                    ),
                    "Сценарий": plan.title,
                    "Платёж / мес.": money(plan.monthly_payment),
                    "Максимальный платёж*": money(plan.peak_monthly_payment),
                    "Финальный остаток": money(plan.final_balance_payment),
                    "Всего выплат": money(plan.total_repayment),
                    "Стоимость финансирования*": money(plan.interest_cost),
                    "Срок": f"{plan.term_months} мес.",
                    "Пауза": f"{plan.grace_months} мес.",
                    "Стресс-сценарии": f"{plan.stress_passes}/{plan.stress_total}",
                }
            )
        st.dataframe(table_rows, width="stretch", hide_index=True)
        st.caption("*Бюджетный лимит проверяет максимальный платёж за месяц, включая финальный остаток. Расчёт без комиссий, страховок и иных договорных платежей.")
        st.download_button(
            "Скачать полный перебор (JSON)",
            data=build_case_report(
                case,
                plans,
                objective,
                budget_share,
                max_extension_months,
                max_grace_months,
            ),
            file_name=f"qaryzflow_{case.case_id.lower()}_schedule_search.json",
            mime="application/json",
            key=f"download_{case.case_id}",
        )


def render_portfolio_page(
    objective: str,
    budget_share: float,
    max_extension_months: int,
    max_grace_months: int,
) -> None:
    summary = portfolio_summary(
        objective,
        budget_share=budget_share,
        max_extension_months=max_extension_months,
        max_grace_months=max_grace_months,
    )
    feasible = [item for item in summary if item["has_feasible_plan"]]
    total_balance = sum(item["balance"] for item in summary)
    if feasible:
        average_reduction = sum(
            item["current_payment"] - (item["recommended_payment"] or 0.0) for item in feasible
        ) / len(feasible)
        average_stress = sum(item["stress_passes"] or 0 for item in feasible) / len(feasible)
    else:
        average_reduction = 0.0
        average_stress = 0.0

    render_header(
        "Портфель учебных кейсов",
        "Проверка поискового алгоритма на шести вымышленных профилях. Сводка меняется вместе с настройками подбора.",
    )
    a, b, c, d = st.columns(4)
    a.metric("Вымышленные кейсы", str(len(summary)))
    b.metric("С доступным графиком", f"{len(feasible)} из {len(summary)}")
    c.metric("Средняя разница платежа", money(average_reduction, compact=True), "текущий минус новый · только подходящие кейсы")
    d.metric("Средние стресс-проверки", f"{average_stress:.1f}/{len(STRESS_SCENARIOS)}", "не вероятность")
    st.caption(f"Суммарный остаток долга в учебных кейсах: {money(total_balance, compact=True)} · цель: {OBJECTIVES[objective]}")

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Результаты по кейсам</div>', unsafe_allow_html=True)
            rows = [
                {
                    "Кейс": item["case_id"],
                    "Регион": item["region"],
                    "Остаток": money(item["balance"], compact=True),
                    "Текущий платёж": money(item["current_payment"], compact=True),
                    "Результат": item["recommended_plan"],
                    "Новый платёж": money(item["recommended_payment"], compact=True),
                    "Максимальный платёж": money(item["recommended_peak_payment"], compact=True),
                    "Всего выплат": money(item["recommended_total_repayment"], compact=True),
                    "Стресс-тесты": (
                        f'{item["stress_passes"]}/{item["stress_total"]}'
                        if item["stress_passes"] is not None
                        else "—"
                    ),
                }
                for item in summary
            ]
            st.dataframe(rows, width="stretch", hide_index=True)
    with right:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Обычный платёж: до и после подбора</div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-caption">Для нового графика показан платёж после паузы; финальный остаток учтён отдельно в таблице.</div>', unsafe_allow_html=True)
            chart_rows = [
                {
                    "Кейс": item["case_id"],
                    "Текущий": item["current_payment"],
                    "Подобранный": item["recommended_payment"],
                }
                for item in feasible
            ]
            if chart_rows:
                st.bar_chart(chart_rows, x="Кейс", y=["Текущий", "Подобранный"], color=["#aebbb6", "#13866d"])
            else:
                st.info("При заданных параметрах доступный график не найден ни для одного кейса.")

    st.write("")
    st.markdown(
        '<div class="callout"><strong>Как читать:</strong> показатели относятся только к шести синтетическим профилям и не описывают банковский портфель. Если график не проходит бюджетный лимит, приложение не считает его рекомендацией. Для банка эффект нужно измерять на разрешённых исторических данных.</div>',
        unsafe_allow_html=True,
    )


def render_optimizer_page(
    case: BorrowerCase,
    objective: str,
    budget_share: float,
    max_extension_months: int,
    max_grace_months: int,
) -> None:
    plans = make_plans(
        case,
        budget_share=budget_share,
        max_extension_months=max_extension_months,
        max_grace_months=max_grace_months,
    )
    feasible_count = sum(plan.is_feasible for plan in plans)
    frontier_count = sum(plan.is_pareto_optimal for plan in plans)

    render_header(
        "Оптимизатор и контроль",
        "Расчёт графиков основан на явных формулах и ограничениях. Вероятность просрочки не моделируется.",
    )
    a, b, c, d = st.columns(4)
    a.metric("Перебрано графиков", str(len(plans)))
    b.metric("Проходят бюджетный лимит", str(feasible_count))
    c.metric("Вариантов на границе Парето", str(frontier_count))
    d.metric("Стресс-проверки", str(len(STRESS_SCENARIOS)), "детерминированные сценарии")

    left, right = st.columns([1, 1], gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Как подбирается график</div>', unsafe_allow_html=True)
            steps = [
                ("01", "Обновить бюджет", "Обычный доход уменьшается на указанный процент; вычитаются обязательные расходы и прочие платежи."),
                ("02", "Перебрать условия", "Проверяются сроки с шагом в один месяц, заданная длина паузы и максимальное продление."),
                ("03", "Отсеять недоступные", f"Остаются планы с платежом не выше {budget_share:.0%} свободного бюджета."),
                ("04", "Показать компромисс", "Сравниваются ежемесячный платёж, сумма выплат и устойчивость к четырём заданным изменениям бюджета."),
            ]
            for number, title, description in steps:
                st.markdown(
                    f'<div class="signal"><span class="signal-mark">{number}</span><span><b>{title}</b><br>{description}</span></div>',
                    unsafe_allow_html=True,
                )
    with right:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Правило выбранной цели</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="panel-caption"><b>{OBJECTIVES[objective]}</b></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="callout">{objective_explanation(objective)}</div>', unsafe_allow_html=True)
            st.markdown(
                """
                <div class="signal"><span class="signal-mark">₸</span><span><b>Аннуитетный платёж</b><br>
                Остаток долга и капитализированные проценты распределяются на срок после паузы по номинальной годовой ставке, делённой на 12.</span></div>
                <div class="signal"><span class="signal-mark">↗</span><span><b>Граница Парето</b><br>
                График сохраняется, если нельзя одновременно уменьшить платёж и общую сумму выплат и при этом пройти больше стресс-проверок.</span></div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    with st.container(border=True):
        st.markdown('<div class="panel-title">Заданные стресс-сценарии</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-caption">Проверяется только соблюдение бюджетного лимита. Результаты не являются шансом дефолта.</div>', unsafe_allow_html=True)
        stress_rows = [
            {
                "Сценарий": name,
                "Снижение дохода": f"{income_shock:.0%}",
                "Рост обязательных расходов": f"{expense_shock:.0%}",
            }
            for name, income_shock, expense_shock in STRESS_SCENARIOS
        ]
        st.dataframe(stress_rows, width="stretch", hide_index=True)

    with st.expander("Стенд на шести вымышленных профилях", expanded=False):
        benchmark_rows = []
        for demo_case in CASES:
            demo_plans = make_plans(
                demo_case,
                budget_share=budget_share,
                max_extension_months=max_extension_months,
                max_grace_months=max_grace_months,
            )
            demo_recommendation = choose_plan(demo_plans, objective)
            benchmark_rows.append(
                {
                    "Кейс": demo_case.case_id,
                    "Всего кандидатов": len(demo_plans),
                    "Подходят по бюджету": sum(plan.is_feasible for plan in demo_plans),
                    "Вариант на границе Парето": sum(plan.is_pareto_optimal for plan in demo_plans),
                    "Выбранный стресс-результат": (
                        f"{demo_recommendation.stress_passes}/{demo_recommendation.stress_total}"
                        if demo_recommendation
                        else "Нет подходящего графика"
                    ),
                }
            )
        st.dataframe(benchmark_rows, width="stretch", hide_index=True)

    st.write("")
    st.markdown(
        """
        <div class="callout">
          <strong>Границы демо.</strong> Основной подбор не использует обученную банковскую модель риска. Отдельная ML-лаборатория обучается только на искусственных примерах и не влияет на этот результат. Не учитываются комиссии, страховки, юридические ограничения, кредитная политика и особенности договоров. Снижение дохода считается от введённого обычного чистого дохода. Все выводы относятся только к вымышленным профилям; сотрудник и правила банка остаются решающими.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ml_lab_page(
    case: BorrowerCase,
    objective: str,
    budget_share: float,
    max_extension_months: int,
    max_grace_months: int,
) -> None:
    render_header(
        "ML-лаборатория",
        "Сравните две модели, проверьте их на более поздней синтетической когорте и посмотрите, как меняется ранжирование графиков.",
    )
    st.warning(
        "Все записи, целевые метки и результаты ниже искусственные. Эта лаборатория показывает процесс проверки ML-модели; её оценки нельзя переносить на клиентов или портфель банка."
    )

    with st.container(border=True):
        st.markdown('<div class="panel-title">Настройки эксперимента</div>', unsafe_allow_html=True)
        data_column, seed_column = st.columns(2)
        rows = data_column.select_slider(
            "Размер искусственной выборки",
            options=(2_000, 4_000, 8_000, 12_000, 16_000),
            value=DEFAULT_ROWS,
            format_func=lambda value: f"{value:,}".replace(",", " "),
        )
        seed = seed_column.number_input(
            "Seed генератора",
            min_value=0,
            max_value=2_147_483_647,
            value=DEFAULT_SEED,
            step=1,
            help="При одинаковом seed синтетические данные и результаты повторяются.",
        )

    model = cached_synthetic_model(rows=int(rows), seed=int(seed))
    selected_model_key = cast(
        str,
        st.selectbox(
            "Модель для ранжирования графиков",
            options=RANKING_MODEL_KEYS,
            format_func=lambda key: RANKING_MODEL_LABELS[key],
            help="Сравнение с детерминированным правилом остаётся доступно для каждой модели.",
        ),
    )
    selected_model = model.get_model(selected_model_key)
    plans = make_plans(
        case,
        budget_share=budget_share,
        max_extension_months=max_extension_months,
        max_grace_months=max_grace_months,
    )
    scored_plans = sorted(
        (
            (score_plan(model, case, plan, selected_model_key), plan)
            for plan in plans
            if plan.is_feasible
        ),
        key=lambda item: (-item[0], item[1].term_months, item[1].grace_months),
    )
    rule_plan = choose_plan(plans, objective)

    a, b, c, d = st.columns(4)
    a.metric("Искусственных примеров", f"{model.rows_total:,}".replace(",", " "))
    b.metric("Обучающая выборка", f"{model.rows_train:,}".replace(",", " "))
    c.metric("Будущая контрольная когорта", f"{model.rows_test:,}".replace(",", " "))
    d.metric("ROC-AUC выбранной модели", f"{selected_model.roc_auc:.3f}", "только симуляция")
    st.caption(
        f"Первые 80% записей используются для обучения, последние 20% — как имитация будущего периода с более напряжёнными условиями. Seed: {model.seed}. Доля метки «успех» меняется с {model.train_success_share:.1%} в обучении до {model.test_success_share:.1%} в контрольной когорте."
    )

    st.subheader("Сравнение качества на отложенном синтетическом периоде")
    comparison_rows = [
        {
            "Модель": item.label,
            "ROC-AUC ↑": f"{item.roc_auc:.3f}",
            "Average precision ↑": f"{item.average_precision:.3f}",
            "Brier ↓": f"{item.brier_score:.3f}",
            "Log loss ↓": f"{item.log_loss_value:.3f}",
        }
        for item in model.models
    ]
    st.dataframe(comparison_rows, width="stretch", hide_index=True)
    st.caption(
        "В таблице есть простой базовый уровень (частота успеха). ROC-AUC оценивает ранжирование; average precision (AP) учитывает точность и полноту положительного класса; Brier и log loss оценивают вероятностные прогнозы."
    )

    left, right = st.columns([1, 1], gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Калибровка выбранной модели</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="panel-caption">Точки группируют записи по близкому синтетическому скору. Идеальная линия означает совпадение среднего скора с долей искусственного успеха.</div>',
                unsafe_allow_html=True,
            )
            calibration_rows = [
                {
                    "Средний скор": point.mean_score,
                    "Доля успеха": point.observed_success,
                    "Идеальная линия": point.mean_score,
                }
                for point in selected_model.calibration
            ]
            st.line_chart(
                calibration_rows,
                x="Средний скор",
                y=["Доля успеха", "Идеальная линия"],
                color=["#13866d", "#9aa9a4"],
            )
            st.caption(f"Brier score: {selected_model.brier_score:.3f} · Log loss: {selected_model.log_loss_value:.3f}.")
    with right:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Проверка по диапазонам нагрузки</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="panel-caption">Сравниваем синтетический скор с результатом симуляции при разном отношении платежа к свободному бюджету. Это диагностика сегментов, не аудит справедливости.</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(
                list(selected_model.burden_segments),
                width="stretch",
                hide_index=True,
            )

    with st.expander("Какие признаки влияли на ROC-AUC выбранной модели", expanded=False):
        importance_rows = [
            {"Признак": name, "Падение ROC-AUC при перемешивании": value}
            for name, value in selected_model.permutation_importance
        ]
        st.bar_chart(
            importance_rows,
            x="Признак",
            y="Падение ROC-AUC при перемешивании",
            color="#13866d",
            horizontal=True,
        )
        st.caption("Перестановочная важность измерена на искусственной контрольной когорте. Она показывает зависимость метрики в симуляции, а не причинность.")

    with st.expander("Как перейти от демо к банковской модели", expanded=False):
        st.markdown(
            "1. Согласовать с банком событие, которое нужно прогнозировать, горизонт и момент принятия решения.\n"
            "2. Подготовить разрешённые признаки, доступные именно на дату решения; исключить утечку будущей информации.\n"
            "3. Сравнить модели на временном holdout и с базовым процессом, проверить калибровку и результат по рабочим сегментам.\n"
            "4. Провести независимую проверку, теневое наблюдение и согласовать мониторинг до любого применения."
        )

    st.write("")
    with st.container(border=True):
        st.markdown('<div class="panel-title">Ранжирование вариантов для выбранного учебного профиля</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-caption">Проценты — выход модели, обученной на искусственных метках. Они не являются реальной вероятностью погашения или рекомендацией банка.</div>',
            unsafe_allow_html=True,
        )
        if scored_plans:
            best_score, best_plan = scored_plans[0]
            left_metric, right_metric = st.columns(2)
            left_metric.metric("Первый по синтетическому скору", best_plan.title, f"скор {best_score:.1%}")
            if rule_plan:
                rule_score = next(score for score, plan in scored_plans if plan.key == rule_plan.key)
                right_metric.metric("Выбор детерминированного правила", rule_plan.title, f"синтетический скор {rule_score:.1%}")

            visible = scored_plans[:8]
            visible_keys = {plan.key for _, plan in visible}
            if rule_plan and rule_plan.key not in visible_keys:
                visible.append(next(item for item in scored_plans if item[1].key == rule_plan.key))
            rank_by_key = {plan.key: rank for rank, (_, plan) in enumerate(scored_plans, start=1)}
            ranking_rows = [
                {
                    "Ранг модели": rank_by_key[plan.key],
                    "Сценарий": plan.title,
                    "Синтетический скор": f"{score:.1%}",
                    "Обычный платёж": money(plan.monthly_payment, compact=True),
                    "Максимальный платёж": money(plan.peak_monthly_payment, compact=True),
                    "Всего выплат": money(plan.total_repayment, compact=True),
                    "Выбран правилом": "Да" if rule_plan and plan.key == rule_plan.key else "",
                }
                for score, plan in visible
            ]
            st.dataframe(ranking_rows, width="stretch", hide_index=True)
        else:
            st.info("Для текущих ограничений нет графиков, проходящих бюджетный лимит; модель не ранжирует недопустимые варианты.")

    with st.expander("Примеры искусственных строк контрольной выборки", expanded=False):
        st.dataframe(list(model.holdout_examples), width="stretch", hide_index=True)

    st.markdown(
        '<div class="callout"><strong>Что означает качество на экране:</strong> ROC-AUC и Brier score измерены на отложенной части той же искусственно сгенерированной выборки. Они показывают, что код обучения и оценки работает на заданной симуляции; это не свидетельство точности, пользы или справедливости модели для клиентов банка.</div>',
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">Q</div><div><div class="brand-name">qaryzflow</div><div class="brand-caption">графики урегулирования</div></div></div>',
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Раздел",
        ["Кейс клиента", "Портфель", "Оптимизатор и контроль", "ML-лаборатория"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown('<div class="sidebar-section">УЧЕБНЫЙ ПРОФИЛЬ</div>', unsafe_allow_html=True)
    selected_case_id = st.selectbox(
        "Демо-кейс",
        options=[case.case_id for case in CASES],
        format_func=lambda value: next(case.label for case in CASES if case.case_id == value),
        label_visibility="collapsed",
    )
    base_case = next(case for case in CASES if case.case_id == selected_case_id)
    st.markdown(
        f'<div class="tiny-note" style="color:#a7c4b9;margin:.2rem 0 1rem">{base_case.region} · {base_case.segment} · вымышленный профиль</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Настройки подбора", expanded=True):
        objective = st.selectbox(
            "Приоритет",
            options=list(OBJECTIVES),
            format_func=lambda key: OBJECTIVES[key],
            help="Сравнение выполняется только среди графиков, прошедших бюджетный лимит.",
        )
        budget_share_pct = st.slider(
            "Лимит платежа от свободного бюджета",
            40,
            100,
            80,
            5,
            help="Графики выше лимита не считаются подходящими.",
        )
        max_extension_months = st.slider("Максимальное продление срока, мес.", 0, 60, 48, 1)
        max_grace_months = st.slider("Максимальная пауза, мес.", 0, 6, 3, 1)

    with st.expander("Данные учебного профиля", expanded=(page == "Кейс клиента")):
        st.markdown('<div class="sidebar-section">Доход и бюджет</div>', unsafe_allow_html=True)
        income = st.number_input(
            "Обычный чистый доход в месяц",
            50_000,
            5_000_000,
            int(base_case.income),
            10_000,
            key=f"income_{base_case.case_id}",
        )
        income_drop_pct = st.slider(
            "Снижение дохода",
            0,
            70,
            int(base_case.income_drop * 100),
            1,
            key=f"drop_{base_case.case_id}",
        )
        st.caption(f"Доход после снижения: {money(income * (1 - income_drop_pct / 100), compact=True)}")
        essential_expenses = st.number_input(
            "Обязательные расходы",
            30_000,
            3_000_000,
            int(base_case.essential_expenses),
            5_000,
            key=f"expenses_{base_case.case_id}",
        )
        other_debt = st.number_input(
            "Платежи по другим долгам",
            0,
            2_000_000,
            int(base_case.other_debt_payment),
            5_000,
            key=f"other_{base_case.case_id}",
        )
        savings = st.number_input(
            "Доступный резерв",
            0,
            20_000_000,
            int(base_case.savings),
            10_000,
            key=f"savings_{base_case.case_id}",
        )
        st.markdown('<div class="sidebar-section">Условия долга</div>', unsafe_allow_html=True)
        balance = st.number_input(
            "Остаток долга",
            50_000,
            50_000_000,
            int(base_case.balance),
            50_000,
            key=f"balance_{base_case.case_id}",
        )
        current_payment = st.number_input(
            "Текущий ежемесячный платёж",
            5_000,
            2_000_000,
            int(base_case.current_payment),
            5_000,
            key=f"payment_{base_case.case_id}",
        )
        annual_rate = st.slider(
            "Номинальная годовая ставка",
            0.0,
            45.0,
            float(base_case.annual_rate),
            0.5,
            key=f"rate_{base_case.case_id}",
        )
        remaining_months = st.slider(
            "Осталось месяцев по договору",
            3,
            84,
            base_case.remaining_months,
            1,
            key=f"term_{base_case.case_id}",
        )
        overdue_months = st.slider(
            "Месяцы текущей просрочки",
            0,
            6,
            base_case.overdue_months,
            1,
            key=f"overdue_{base_case.case_id}",
        )

    scenario_case = replace(
        base_case,
        income=input_number(income, base_case.income),
        income_drop=float(income_drop_pct) / 100.0,
        balance=input_number(balance, base_case.balance),
        current_payment=input_number(current_payment, base_case.current_payment),
        other_debt_payment=input_number(other_debt, base_case.other_debt_payment),
        essential_expenses=input_number(essential_expenses, base_case.essential_expenses),
        savings=input_number(savings, base_case.savings),
        overdue_months=int(overdue_months),
        remaining_months=int(remaining_months),
        annual_rate=input_number(annual_rate, base_case.annual_rate),
    )
    budget_share = float(budget_share_pct) / 100.0
    st.divider()
    st.markdown('<span class="status-pill"><span class="status-dot"></span>Локальный демо-режим</span>', unsafe_allow_html=True)
    st.markdown(
        '<div class="tiny-note" style="color:#a7c4b9;margin-top:12px">Внешние сервисы и реальные клиентские данные не используются.</div>',
        unsafe_allow_html=True,
    )


if page == "Кейс клиента":
    render_case_page(
        scenario_case,
        objective,
        budget_share,
        max_extension_months,
        max_grace_months,
    )
elif page == "Портфель":
    render_portfolio_page(
        objective,
        budget_share,
        max_extension_months,
        max_grace_months,
    )
elif page == "Оптимизатор и контроль":
    render_optimizer_page(
        scenario_case,
        objective,
        budget_share,
        max_extension_months,
        max_grace_months,
    )
else:
    render_ml_lab_page(
        scenario_case,
        objective,
        budget_share,
        max_extension_months,
        max_grace_months,
    )
