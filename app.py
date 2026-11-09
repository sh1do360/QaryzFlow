from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import cast

import altair as alt
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
    monthly_budget,
    monthly_schedule,
    objective_explanation,
    payment_within_limit,
    portfolio_summary,
    recommendation_reason,
    sort_plans,
    stress_details,
)
from qadam_ml_demo import (
    DEFAULT_ROWS,
    DEFAULT_SEED,
    RANKING_MODEL_KEYS,
    RANKING_MODEL_LABELS,
    SyntheticModelDemo,
    rank_feasible_plans,
    score_sensitivity,
    train_synthetic_model,
)
from qadam_report import build_case_report, build_review_summary
from qadam_synthetic import MAX_SEED, OBSERVED_PAYMENTS


st.set_page_config(
    page_title="QaryzFlow · Реструктуризация долга",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="auto",
)


@st.cache_resource(max_entries=8, show_spinner="Симулируем платежи и обучаем учебные модели…")
def cached_synthetic_model(rows: int, seed: int) -> SyntheticModelDemo:
    return train_synthetic_model(rows=rows, seed=seed)


def money(value: float | None, compact: bool = False, *, decimals: int = 0) -> str:
    if value is None:
        return "—"
    if decimals:
        return f"{value:,.{decimals}f} ₸".replace(",", " ").replace(".", ",")
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


def change_from_current(new_value: float, current_value: float) -> str:
    difference = round(new_value - current_value)
    if difference == 0:
        return "Без изменения"
    direction = "меньше" if difference < 0 else "больше"
    return f"На {money(abs(difference))} {direction}"


st.markdown(
    """
    <style>
      :root { --ink:#132321; --muted:#6d7d79; --line:#e3ebe6; --green:#087f69; --mint:#dff5ee; --surface:#fff; --canvas:#f3f7f5; }
      html, body, [class*="css"] { font-family:'Segoe UI','Aptos',sans-serif; }
      .stApp { background:radial-gradient(ellipse at 8% 0%,rgba(208,239,225,.52),transparent 34%),var(--canvas);color:var(--ink); }
      [data-testid="stHeader"] { background:rgba(243,247,245,.88);backdrop-filter:blur(12px); }
      [data-testid="stSidebar"] { background:linear-gradient(180deg,#102d29 0%,#123630 100%);border-right:0; }
      [data-testid="stSidebar"] * { color:#eef7f3; }
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
      .panel-title { font-weight:780;color:var(--ink);font-size:1rem;margin-bottom:4px; }
      .panel-caption { font-size:.78rem;color:var(--muted);line-height:1.5;margin-bottom:14px; }
      [data-testid="stVerticalBlockBorderWrapper"] { background:#fff;border:1px solid var(--line) !important;border-radius:18px !important;box-shadow:0 8px 24px rgba(19,48,40,.045); }
      .signal { display:flex;align-items:flex-start;gap:9px;color:#40524e;font-size:.81rem;margin:10px 0; }
      .signal-mark { flex:0 0 19px;width:19px;height:19px;border-radius:50%;background:#e5f4ec;color:#087f69;display:grid;place-items:center;font-size:.7rem;font-weight:800; }
      .callout { border-radius:13px;background:#eef7f3;border:1px solid #dcece4;padding:13px 15px;color:#405a51;font-size:.8rem;line-height:1.55; }
      .callout strong { color:#164d3f; }
      .stButton > button,[data-testid="stDownloadButton"] > button { border-radius:11px;border:1px solid #d9e6e0;font-weight:750;transition:transform .15s ease,box-shadow .15s ease; }
      .stButton > button:hover,[data-testid="stDownloadButton"] > button:hover { border-color:#9fcab8;box-shadow:0 5px 14px rgba(20,84,67,.1);transform:translateY(-1px); }
      [data-testid="stMetric"] { background:#fff;border:1px solid var(--line);border-radius:16px;padding:16px 17px;box-shadow:0 6px 20px rgba(19,48,40,.035); }
      [data-testid="stMetricLabel"] { color:#70817c;font-size:.73rem;font-weight:700; }
      [data-testid="stMetricValue"] { color:var(--ink);font-weight:800;letter-spacing:-.045em; }
      [data-testid="stDataFrame"] { border:1px solid var(--line);border-radius:14px;overflow:hidden;background:#fff; }
      @media(max-width:800px) { .block-container { padding:1rem 1rem 2rem; } h1 {font-size:1.65rem !important;} }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_header(title: str, subtitle: str) -> None:
    st.caption("QaryzFlow · Учебный режим")
    st.title(title, anchor=False)
    st.write(subtitle)


def render_case_ml_comparison(
    case: BorrowerCase, plans: list[Plan], recommended: Plan | None
) -> Plan | None:
    with st.container(border=True):
        enabled = st.toggle(
            "Сравнить с ML в симуляции",
            key="case_ml_comparison",
            disabled=recommended is None,
        )
        if recommended is None:
            st.caption("Сравнение доступно, когда хотя бы один график проходит бюджетный лимит.")
            return None
        if not enabled:
            st.caption("Посмотрите, какой из допустимых графиков выбрала бы модель на искусственных данных.")
            return None

        model_key = cast(
            str,
            st.selectbox(
                "Модель для сравнения",
                options=RANKING_MODEL_KEYS,
                format_func=RANKING_MODEL_LABELS.__getitem__,
                key="case_ml_model",
            ),
        )
        demo = cached_synthetic_model(DEFAULT_ROWS, DEFAULT_SEED)
        ranked = rank_feasible_plans(demo, case, plans, model_key)
        ml_score, ml_plan = ranked[0]
        rule_score = next(score for score, plan in ranked if plan.key == recommended.key)
        st.caption("Баллы отражают искусственный исход в симуляции. Выбор по бюджету и заданной цели показан отдельно.")
        st.dataframe(
            [
                {
                    "Подход": label,
                    "График": plan.title,
                    "Максимальный платёж": money(plan.peak_monthly_payment),
                    "Всего выплат": money(plan.total_repayment),
                    "Стресс-проверки": f"{plan.stress_passes}/{plan.stress_total}",
                    "Учебный балл (0–100)": round(100 * score, 1),
                }
                for label, plan, score in (
                    ("По правилам", recommended, rule_score),
                    ("По ML в симуляции", ml_plan, ml_score),
                )
            ],
            width="stretch",
            hide_index=True,
        )
        if ml_plan.key == recommended.key:
            st.caption("Оба подхода выбрали один график.")
        else:
            payment_delta = f"{ml_plan.peak_monthly_payment - recommended.peak_monthly_payment:+,.0f}".replace(",", " ")
            total_delta = f"{ml_plan.total_repayment - recommended.total_repayment:+,.0f}".replace(",", " ")
            st.caption(
                f"Разница ML-варианта к выбору по правилам: {payment_delta} ₸ к максимальному платежу, "
                f"{total_delta} ₸ к общей сумме выплат."
            )
            changes = score_sensitivity(demo, case, recommended, ml_plan, model_key)
            if any(abs(delta) >= 0.0005 for _, delta in changes):
                with st.expander("Какие изменения повлияли на учебный балл"):
                    st.caption("По одному подставляем параметры ML-варианта в график по правилам и сравниваем баллы модели.")
                    st.dataframe(
                        [
                            {"Параметр": label, "Изменение балла": round(100 * delta, 2)}
                            for label, delta in changes
                        ],
                        width="stretch",
                        hide_index=True,
                    )
        return ml_plan


def choose_review_plan(
    case: BorrowerCase, eligible: list[Plan], recommended: Plan, objective: str, budget_share: float
) -> Plan:
    by_key = {plan.key: plan for plan in eligible}
    selector_key = f"review_plan_{case.case_id}"
    if st.session_state.get(selector_key, "auto") not in {"auto", *by_key}:
        st.session_state[selector_key] = "auto"
        st.info("Выбранный ранее график вышел за пределы текущих условий. Показано предложение по заданной цели.")

    current_key = st.session_state.get(selector_key, "auto")
    st.caption("Сравните три предложения или найдите другой допустимый график в списке ниже.")
    cost_is_current = False
    for column, (goal, label) in zip(
        st.columns(3),
        (("affordable", "Меньше платёж"), ("cost", "Меньше сумма выплат"), ("robust", "Больше стресс-проверок")),
        strict=True,
    ):
        plan = choose_plan(eligible, goal)
        if plan is None:
            continue
        keep_current = goal == "cost" and plan.key == "current"
        cost_is_current = cost_is_current or keep_current
        no_stress_passes = goal == "robust" and plan.stress_passes == 0
        with column.container(border=True):
            st.markdown(f"**{'Стресс-проверки' if no_stress_passes else label}**")
            st.caption(plan.title)
            st.metric("Максимум за месяц", money(plan.peak_monthly_payment))
            st.write(f"Всего выплат **{money(plan.total_repayment)}**")
            st.caption(f"Стресс-проверки {plan.stress_passes} из {plan.stress_total}")
            if keep_current:
                st.caption("Текущий график проходит бюджетный лимит и уже даёт минимальную сумму выплат.")
            if no_stress_passes:
                worst = max(stress_details(case, plan, budget_share), key=lambda check: check.shortfall)
                st.caption("Ни один доступный график не прошёл ни одной стресс-проверки.")
                st.caption(f"В этом варианте наибольшая нехватка — {money(worst.shortfall)} в месяце {worst.shortfall_month}. Сценарий «{worst.name}».")
            active = current_key == plan.key
            action = "Оставить текущий график" if keep_current else "Выбрать график"
            st.button(
                "Выбран для сравнения" if active else action,
                key=f"{selector_key}_{goal}",
                on_click=st.session_state.__setitem__,
                args=(selector_key, plan.key),
                disabled=active,
                icon=":material/check:" if active else None,
                width="stretch",
            )

    if "current" in by_key and not cost_is_current:
        st.button(
            "Оставить текущий график" if current_key != "current" else "Текущий график выбран",
            key=f"{selector_key}_keep_current",
            on_click=st.session_state.__setitem__,
            args=(selector_key, "current"),
            disabled=current_key == "current",
            help="Текущий график укладывается в бюджетный лимит.",
        )

    labels = {"auto": f"По цели «{OBJECTIVES[objective]}» — {recommended.title}"}
    labels.update(
        {
            plan.key: f"{plan.title} · {money(plan.peak_monthly_payment)} максимум · {money(plan.total_repayment)} всего"
            for plan in eligible
        }
    )
    selection = st.selectbox(
        "График для обсуждения",
        options=list(labels),
        format_func=labels.__getitem__,
        key=selector_key,
        persist_state="session",
        help="Строка «По цели» обновляется автоматически. Можно выбрать любой допустимый график вручную.",
    )
    return by_key.get(cast(str, selection), recommended)


def render_payment_calendar(
    case: BorrowerCase, budget_share: float, schedule_plans: list[tuple[str, Plan]]
) -> None:
    term = max(plan.term_months for _, plan in schedule_plans)
    recovery_end = case.income_recovery_wait_months + case.income_recovery_months
    preview = min(term, max(12, recovery_end))
    budget = monthly_budget(case, budget_share, term)
    series = {label: monthly_schedule(case, plan, term) for label, plan in schedule_plans}
    series["Лимит платежа"] = [month.payment_limit for month in budget]
    rows = [
        {
            "Месяц": month.month,
            "График": label,
            "Сумма": payment,
            "Лимит": month.payment_limit,
            "Превышение": 0.0 if payment_within_limit(payment, month.payment_limit) else payment - month.payment_limit,
        }
        for label, payments in series.items()
        for month, payment in zip(budget, payments, strict=True)
    ]
    over_limit = [row for row in rows if row["Превышение"] > 0]
    with st.container(border=True):
        st.subheader("Помесячный бюджет и платежи", anchor=False)
        full = st.toggle("Показать весь срок", key=f"calendar_full_{case.case_id}", disabled=preview == term)
        horizon = term if full else preview
        st.caption(f"Месяцы 1–{horizon}. В паузу начисленные проценты добавляются к долгу.")
        colors = {
            "Текущий график": "#63766e",
            "График для обсуждения": "#13866d",
            "Вариант для анализа": "#b86053",
            "Вариант из ML-симуляции": "#8c5bb0",
            "Лимит платежа": "#c46b16",
        }
        visible = [row for row in rows if row["Месяц"] <= horizon]
        base = alt.Chart(alt.InlineData(values=visible)).encode(
            x=alt.X("Месяц:Q", axis=alt.Axis(tickMinStep=1, format="d"), scale=alt.Scale(zero=False)),
            y=alt.Y("Сумма:Q", title="Сумма, ₸"),
        )
        tooltips = [alt.Tooltip("Месяц:Q", format="d"), "График:N", alt.Tooltip("Сумма:Q", format=",.2f")]
        lines = base.mark_line(strokeWidth=2.2, point=alt.OverlayMarkDef(size=20)).encode(
            color=alt.Color("График:N", title=None, scale=alt.Scale(domain=list(series), range=[colors[label] for label in series])),
            strokeDash=alt.condition(alt.datum["График"] == "Лимит платежа", alt.value([6, 4]), alt.value([1, 0])),
            tooltip=tooltips,
        )
        breaches = base.transform_filter(alt.datum["Превышение"] > 0).mark_point(
            shape="diamond", size=95, filled=True, color="#b9382e", stroke="#ffffff", strokeWidth=1,
        ).encode(
            tooltip=[*tooltips, alt.Tooltip("Лимит:Q", format=",.2f"), alt.Tooltip("Превышение:Q", format=",.2f")],
        )
        st.altair_chart(lines + breaches, width="stretch", alt="Платежи и лимит по месяцам. Ромб отмечает превышение лимита.")
        if over_limit:
            worst = max(over_limit, key=lambda row: row["Превышение"])
            st.caption(
                f"Ромбы отмечают платежи выше лимита. Наибольшее превышение за весь срок — "
                f"{money(worst['Превышение'], decimals=2)} в месяце {worst['Месяц']}. {worst['График']}."
            )
            if any(row["Месяц"] > horizon for row in over_limit):
                st.info("Некоторые платежи выше лимита за пределами этого интервала. Включите «Показать весь срок».")

    with st.expander("Полный календарь выплат"):
        for label, plan in schedule_plans:
            st.markdown(f"**{label}**")
            st.dataframe(
                [
                    {
                        "Месяц": row.month,
                        "Доход": money(month.income),
                        "Лимит платежа": money(month.payment_limit),
                        "Долг на начало": money(row.opening_balance),
                        "Платёж": money(row.payment),
                        "Выше лимита": (
                            "—" if payment_within_limit(row.payment, month.payment_limit)
                            else money(row.payment - month.payment_limit, decimals=2)
                        ),
                        "Начислено процентов": money(row.accrued_interest),
                        "Уплачено процентов": money(row.interest_paid),
                        "Погашено долга": money(row.principal_repaid),
                        "Капитализировано": money(row.capitalized_interest),
                        "Долг на конец": money(row.closing_balance),
                    }
                    for row, month in zip(amortization_schedule(case, plan), budget)
                ],
                width="stretch",
                hide_index=True,
                alt=f"Календарь выплат. {label}",
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
    current = plans[0]
    eligible = [plan for plan in sort_plans(plans, objective) if plan.is_feasible]
    recommended = eligible[0] if eligible else None
    cash = free_cash(case)
    st.subheader("2. Выберите график", anchor=False)
    cash_metric, limit_metric, plan_count = st.columns(3)
    cash_metric.metric(
        "Свободно в первый месяц",
        money(cash),
        f"Доход {money(effective_income(case))}",
        delta_color="off",
        delta_arrow="off",
        help="Доход после снижения за вычетом обязательных расходов и других долгов.",
    )
    limit_metric.metric(
        "Лимит платежа в первый месяц",
        money(max(0.0, cash) * budget_share),
        f"{budget_share:.0%} свободного бюджета",
        delta_color="off",
        delta_arrow="off",
    )
    plan_count.metric("Подходящие графики", str(len(eligible)), f"из {len(plans)} рассчитанных", delta_color="off", delta_arrow="off")

    selected = None
    if recommended is not None:
        selected = choose_review_plan(case, eligible, recommended, objective, budget_share)
        shown_plan = selected
        shown_label = "График для обсуждения"
    else:
        shown_plan = min(
            (plan for plan in plans if plan.key != current.key),
            key=lambda plan: (plan.feasibility_gap, plan.peak_monthly_payment, plan.total_repayment),
            default=current,
        )
        shown_label = "Вариант для анализа"
        st.warning("При этих условиях подходящего графика нет. Можно изменить исходные данные или границы поиска в блоке «Изменить условия».")

    st.subheader("3. Сравните с текущим", anchor=False)
    with st.container(border=True):
        st.markdown(f"**{shown_label} · {shown_plan.title}**")
        st.caption("Изменения ниже указаны относительно текущего графика.")
        peak, total, stress = st.columns(3)
        peak.metric(
            "Максимальный платёж",
            money(shown_plan.peak_monthly_payment),
            change_from_current(shown_plan.peak_monthly_payment, current.peak_monthly_payment),
            delta_color="off",
            delta_arrow="off",
        )
        total.metric(
            "Всего выплат",
            money(shown_plan.total_repayment),
            change_from_current(shown_plan.total_repayment, current.total_repayment),
            delta_color="off",
            delta_arrow="off",
        )
        stress.metric("Пройдено стресс-проверок", f"{shown_plan.stress_passes} из {shown_plan.stress_total}")
        if selected is not None:
            st.caption(recommendation_reason(case, selected, budget_share)[0])
            st.download_button(
                "Скачать краткое сравнение",
                data=build_review_summary(case, current, selected, budget_share),
                file_name=f"qaryzflow_{case.case_id.lower()}_review.html",
                mime="text/html",
                on_click="ignore",
                icon=":material/download:",
                help="Откройте скачанный файл в браузере для просмотра или печати.",
            )
        else:
            st.caption(
                f"Ближайший к лимиту новый график для разбора. "
                f"В месяце {shown_plan.budget_check_month} не хватает {money(shown_plan.feasibility_gap)}."
            )

    calendar_tab, stress_tab, plans_tab, ml_tab = st.tabs(["Календарь выплат", "Стресс-проверки", "Все варианты", "ML-сравнение"])
    with ml_tab:
        ml_plan = render_case_ml_comparison(case, plans, recommended)

    with calendar_tab:
        with st.expander("Текущий и выбранный графики · все показатели"):
            st.table(
                [
                    {"Показатель": label, "Текущий график": before, shown_label: after}
                    for label, before, after in (
                        ("Срок", f"{current.term_months} мес.", f"{shown_plan.term_months} мес."),
                        ("Пауза", f"{current.grace_months} мес.", f"{shown_plan.grace_months} мес."),
                        ("Обычный платёж", money(current.monthly_payment), money(shown_plan.monthly_payment)),
                        ("Максимальный платёж", money(current.peak_monthly_payment), money(shown_plan.peak_monthly_payment)),
                        ("Всего выплат", money(current.total_repayment), money(shown_plan.total_repayment)),
                    )
                ],
                hide_index=True,
                alt="Платежи и стоимость текущего и рассматриваемого графиков",
            )
        schedule_plans = [("Текущий график", current)]
        if shown_plan.key != current.key:
            schedule_plans.append((shown_label, shown_plan))
        if ml_plan and ml_plan.key not in {plan.key for _, plan in schedule_plans}:
            schedule_plans.append(("Вариант из ML-симуляции", ml_plan))
        render_payment_calendar(case, budget_share, schedule_plans)

    with stress_tab:
        st.caption(f"{shown_plan.title}. Проверяем платёж при снижении дохода или росте расходов.")
        st.table(
            [
                {
                    "Сценарий": check.name,
                    "Результат": "Проходит" if check.passed else "Не проходит",
                    "Наибольшая нехватка": (
                        "—" if check.passed else f"Месяц {check.shortfall_month} · {money(check.shortfall)}"
                    ),
                }
                for check in stress_details(case, shown_plan, budget_share)
            ],
            hide_index=True,
            alt="Результаты четырёх стресс-проверок по рассматриваемому графику",
        )

    with plans_tab:
        st.caption("Полный отчёт включает исходные условия, все рассчитанные варианты и помесячный бюджет.")
        st.download_button(
            "Скачать полный перебор (JSON)",
            data=partial(
                build_case_report, case, plans, objective, budget_share, max_extension_months, max_grace_months,
                review_plan_key=selected.key if selected else None,
            ),
            file_name=f"qaryzflow_{case.case_id.lower()}_schedule_search.json",
            mime="application/json",
            key=f"download_{case.case_id}",
            on_click="ignore",
            icon=":material/download:",
        )
        if eligible:
            st.dataframe(
                [
                    {
                        "График": plan.title,
                        "Макс. платёж, ₸": round(plan.peak_monthly_payment),
                        "Всего выплат, ₸": round(plan.total_repayment),
                        "Стресс-проверки": f"{plan.stress_passes}/{plan.stress_total}",
                        "Лучший компромисс": "Да" if plan.is_pareto_optimal else "—",
                        "Для обсуждения": "Выбран" if selected and plan.key == selected.key else "—",
                    }
                    for plan in eligible
                ],
                width="stretch",
                hide_index=True,
                alt="Все графики, прошедшие помесячную бюджетную проверку",
            )
        else:
            st.info("При этих условиях допустимых графиков нет.")


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
        '<div class="callout"><strong>Как читать:</strong> показатели относятся к учебным профилям. В сводке показаны варианты, прошедшие бюджетный лимит для заданных условий.</div>',
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
                ("01", "Рассчитать бюджет по месяцам", "Доход меняется по сценарию восстановления. Из него вычитаются обязательные расходы и прочие платежи."),
                ("02", "Перебрать условия", "Проверяются сроки с шагом в один месяц, заданная длина паузы и максимальное продление."),
                ("03", "Отсеять недоступные", f"В каждом месяце платёж должен укладываться в {budget_share:.0%} свободного бюджета этого месяца."),
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

    with st.expander("Стенд на вымышленных профилях", expanded=False):
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
          <strong>Границы демо.</strong> Основной подбор выполняется по формулам и заданной цели. Сравнение с ML в карточке и лаборатории использует искусственные примеры. Расчёт упрощает договор до долга, ставки, срока и платежей. Снижение дохода считается от введённого обычного чистого дохода. Выводы относятся к вымышленным профилям.
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
            max_value=MAX_SEED,
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
    scored_plans = rank_feasible_plans(model, case, plans, selected_model_key)
    rule_plan = choose_plan(plans, objective)

    a, b, c, d = st.columns(4)
    a.metric("Искусственных примеров", f"{model.rows_total:,}".replace(",", " "))
    b.metric("Обучающая выборка", f"{model.rows_train:,}".replace(",", " "))
    c.metric("Будущая контрольная когорта", f"{model.rows_test:,}".replace(",", " "))
    d.metric("ROC-AUC выбранной модели", f"{selected_model.roc_auc:.3f}", "только симуляция")
    st.caption(
        f"Доля графиков без пропусков платежей — {model.train_success_share:.1%} в обучении и {model.test_success_share:.1%} в поздней проверке. Seed {model.seed}."
    )
    with st.expander("Как получены данные и исходы"):
        st.write(
            f"Каждая запись содержит вымышленный бюджет и рассчитанный из долга, ставки, срока и паузы платёж. "
            f"В симуляции меняются доходы, расходы и время начала восстановления. Успех означает, что заёмщик внёс "
            f"первые {OBSERVED_PAYMENTS} плановых платежей полностью и вовремя, либо все платежи, если их меньше. "
            "Нехватка в платёжном месяце расходует резерв. Поздние 20% записей имитируют период с меньшими резервами и более частыми отклонениями бюджета."
        )
        st.dataframe(
            list(model.scenario_summary),
            width="stretch",
            hide_index=True,
            alt="Состав обучающей и поздней выборок по траектории дохода",
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
        "Сравните полные модели с двумя ориентирами. Один учитывает только долю платежа в бюджете, другой выдаёт одну и ту же долю успеха для всех графиков. ROC-AUC и AP оценивают ранжирование, Brier и log loss — точность учебных вероятностных оценок."
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
            "4. Независимо проверить качество модели и согласовать мониторинг до применения."
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

    with st.expander("Примеры графиков и исходов в контрольной выборке", expanded=False):
        st.caption("Месяц первого пропуска считается от начала графика. Нехватка показывает неоплаченную часть платежа в этом месяце.")
        st.dataframe(list(model.holdout_examples), width="stretch", hide_index=True, alt="Примеры рассчитанных платежей и пропусков в поздней симуляции")

    st.markdown(
        '<div class="callout"><strong>Что означает качество на экране:</strong> ROC-AUC и Brier score измерены на отложенной части той же искусственно сгенерированной выборки. Они показывают, что код обучения и оценки работает на заданной симуляции; это не свидетельство точности, пользы или справедливости модели для клиентов банка.</div>',
        unsafe_allow_html=True,
    )


def reset_case_inputs(case_id: str) -> None:
    fields = (
        "income", "expenses", "other", "savings", "drop", "recovery", "recovery_wait",
        "balance", "payment", "rate", "term", "overdue", "review_plan", "calendar_full",
    )
    for field in fields:
        st.session_state.pop(f"{field}_{case_id}", None)


def render_workspace_inputs(primary: bool) -> tuple[BorrowerCase, str, float, int, int]:
    cases = {case.case_id: case for case in CASES}
    if primary:
        st.subheader("1. Выберите кейс и условия", anchor=False)
    case_id = st.selectbox(
        "Учебный кейс",
        options=list(cases),
        index=list(cases).index("Q-1176"),
        format_func=lambda key: cases[key].label,
        key="case_selector",
    )
    base = cases[case_id]
    st.caption(f"{base.region} · {base.segment} · вымышленный профиль")
    if primary and base.demo_note:
        st.caption(f"Подсказка к исходному кейсу. {base.demo_note}")

    settings = st.expander("Изменить условия") if primary else st.container()
    with settings:
        st.caption("После изменения значения расчёт обновляется сразу.")
        budget_tab, loan_tab, search_tab = st.tabs(["Доход и расходы", "Кредит", "Параметры подбора"])
        with budget_tab:
            amounts, recovery = st.columns(2)
            with amounts:
                st.markdown("**Ежемесячный бюджет**")
                income = st.number_input(
                    "Доход до снижения, ₸", 50_000, 5_000_000, int(base.income), 10_000,
                    key=f"income_{case_id}", persist_state="session",
                )
                essential_expenses = st.number_input(
                    "Обязательные расходы, ₸", 30_000, 3_000_000, int(base.essential_expenses), 5_000,
                    key=f"expenses_{case_id}", persist_state="session",
                )
                other_debt = st.number_input(
                    "Платежи по другим долгам, ₸", 0, 2_000_000, int(base.other_debt_payment), 5_000,
                    key=f"other_{case_id}", persist_state="session",
                )
                savings = st.number_input(
                    "Денежный резерв, ₸", 0, 20_000_000, int(base.savings), 10_000,
                    key=f"savings_{case_id}", persist_state="session",
                    help="Резерв учитывается в учебной ML-симуляции.",
                )
            with recovery:
                st.markdown("**Как меняется доход**")
                income_drop_pct = st.slider(
                    "Снижение дохода", 0, 70, round(base.income_drop * 100), 1,
                    format="%d%%", key=f"drop_{case_id}", persist_state="session",
                )
                first_income = input_number(income, base.income) * (1 - income_drop_pct / 100)
                st.caption(f"Доход в первый месяц — {money(first_income)}.")
                recovery_months = st.slider(
                    "За сколько месяцев восстановится доход", 0, 12, base.income_recovery_months, 1,
                    key=f"recovery_{case_id}", persist_state="session",
                    help="При 0 доход остаётся сниженным на весь срок.",
                )
                recovery_wait_months = st.slider(
                    "Месяцев со сниженным доходом до начала роста", 1, 12, base.income_recovery_wait_months, 1,
                    key=f"recovery_wait_{case_id}", persist_state="session",
                    disabled=recovery_months == 0 or income_drop_pct == 0,
                )

        with loan_tab:
            debt, term = st.columns(2)
            with debt:
                balance = st.number_input(
                    "Остаток долга, ₸", 50_000, 50_000_000, int(base.balance), 50_000,
                    key=f"balance_{case_id}", persist_state="session",
                )
                current_payment = st.number_input(
                    "Текущий платёж в месяц, ₸", 5_000, 2_000_000, int(base.current_payment), 5_000,
                    key=f"payment_{case_id}", persist_state="session",
                )
                annual_rate = st.slider(
                    "Номинальная годовая ставка", 0.0, 45.0, float(base.annual_rate), 0.5,
                    format="%.1f%%", key=f"rate_{case_id}", persist_state="session",
                )
            with term:
                remaining_months = st.slider(
                    "Осталось месяцев по договору", 3, 84, base.remaining_months, 1,
                    key=f"term_{case_id}", persist_state="session",
                )
                overdue_months = st.slider(
                    "Месяцев текущей просрочки", 0, 6, base.overdue_months, 1,
                    key=f"overdue_{case_id}", persist_state="session",
                )

        with search_tab:
            limits, timing = st.columns(2)
            with limits:
                objective = st.selectbox(
                    "Предложение по умолчанию",
                    options=list(OBJECTIVES),
                    format_func=OBJECTIVES.__getitem__,
                    key="schedule_objective",
                    help="Этот приоритет применяется, пока график не выбран вручную.",
                )
                budget_share_pct = st.slider(
                    "Доля свободного бюджета на платёж", 40, 100, 80, 5, format="%d%%",
                    key="budget_share",
                    help="С этой долей сравнивается платёж в каждом месяце.",
                )
            with timing:
                max_extension_months = st.slider(
                    "На сколько месяцев можно продлить срок", 0, 60, 48, 1, key="max_extension_months",
                    help="Продление прибавляется к оставшемуся сроку договора.",
                )
                max_grace_months = st.slider(
                    "Максимальная пауза, месяцев", 0, 6, 3, 1,
                    help="Во время паузы платежей нет, а проценты добавляются к долгу.",
                )

        st.button(
            "Вернуть исходные данные кейса",
            on_click=reset_case_inputs,
            args=(case_id,),
            key=f"reset_case_{case_id}",
            help="Сбросит правки бюджета, кредита и ручной выбор для этого кейса.",
        )

    scenario_case = replace(
        base,
        income=input_number(income, base.income),
        income_drop=float(income_drop_pct) / 100,
        income_recovery_months=int(recovery_months),
        income_recovery_wait_months=int(recovery_wait_months),
        balance=input_number(balance, base.balance),
        current_payment=input_number(current_payment, base.current_payment),
        other_debt_payment=input_number(other_debt, base.other_debt_payment),
        essential_expenses=input_number(essential_expenses, base.essential_expenses),
        savings=input_number(savings, base.savings),
        overdue_months=int(overdue_months),
        remaining_months=int(remaining_months),
        annual_rate=input_number(annual_rate, base.annual_rate),
    )
    if income_drop_pct > 0:
        if recovery_months == 0:
            st.caption("По сценарию доход остаётся сниженным на весь срок.")
        else:
            st.caption(
                f"По сценарию доход начинает расти в месяце {recovery_wait_months + 1} "
                f"и восстанавливается в месяце {recovery_wait_months + recovery_months}."
            )
    return scenario_case, cast(str, objective), float(budget_share_pct) / 100, max_extension_months, max_grace_months


PAGE_HEADERS = {
    "Подбор графика": ("Подбор графика", "Сравните варианты реструктуризации по платежу, общей сумме и запасу бюджета."),
    "Проверка проекта": ("Проверка проекта", "Посмотрите сводку кейсов, правила расчёта и результаты учебного ML-эксперимента."),
}

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">Q</div><div><div class="brand-name">qaryzflow</div><div class="brand-caption">графики урегулирования</div></div></div>',
        unsafe_allow_html=True,
    )
    page = st.radio("Раздел", list(PAGE_HEADERS), key="page")
    st.caption("Локальный режим · вымышленные данные")

render_header(*PAGE_HEADERS[page])
if page == "Подбор графика":
    render_case_page(*render_workspace_inputs(primary=True))
else:
    section = st.segmented_control(
        "Что проверить",
        ["Портфель", "Как считается", "ML-лаборатория"],
        default="Портфель",
        required=True,
        key="project_section",
        persist_state="session",
        width="stretch",
    )
    with st.expander("Исходные данные и настройки"):
        if section == "Портфель":
            st.caption("Для портфеля применяются общие параметры подбора. У каждого профиля свой исходный бюджет.")
        workspace = render_workspace_inputs(primary=False)

    if section == "Портфель":
        render_portfolio_page(*workspace[1:])
    elif section == "Как считается":
        render_optimizer_page(*workspace)
    else:
        render_ml_lab_page(*workspace)
