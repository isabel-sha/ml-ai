import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from datetime import datetime, timedelta
import plotly.express as px
import streamlit.components.v1 as components
import numpy as np
import holidays
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_poisson_deviance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.base import clone
import xgboost as xgb
import time
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import plotly.io as pio


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="ReportNow", layout="wide")

# ---------------- SESSION STATE ----------------
if "datasets" not in st.session_state:
    st.session_state.datasets = {}

if "selected_tab" not in st.session_state:
    st.session_state.selected_tab = "Data"

# ---------------- STANDARDIZATION & GROUP MAPPING ----------------
STANDARDIZATION_RULES = {
    "incident": {"close": "closed", "resolve": "resolved", "opened": "opened",
                 "update": "updated", "create": "created", "department": "department",
                 "contact": "channel", "caller": "requested_for"},
    "request": {"close": "closed", "request": "request", "state": "state"},
    "survey": {"score": "score", "response": "response"},
}

group_mapping = {
    "UITS CatNet 2.0 Support": "Cloud",
    "UITS Cloud and Open Systems": "Cloud",
    "UITS Infra UNIX": "Cloud",
    "UITS Managed Services": "Cloud",
    "UITS MCS/CCI": "Cloud",
    "UITS MS Architecture": "Cloud",
    "UITS WorkSpaces": "Cloud",
    "Employee Engagement Council": "Cloud",
    "UITS Cloud Research Services": "Research",
    "UITS HPC Infra": "Research",
    "UITS HPC Cloud": "Research",
    "UITS CRRSP Team": "Research",
    "UITS Faculty Portfolio": "Research",
    "UITS HPC Consulting": "Research",
    "UITS HPC Statistics Consulting": "Research",
    "UITS Data Center": "Research",
    "UITS ITSM": "ITSM",
    "UITS Contact Center Services": "ITSM",
    "UITS ASITS": "PM and Analysts",
}

# ---------------- FUNCTIONS ----------------


def find_id_column(df):
    for col in df.columns:
        if df[col].dtype == object:
            if df[col].astype(str).str.match(r"^[A-Z]+").any():
                return col
    return None


def detect_dataset_type(df):
    id_col = find_id_column(df)
    if not id_col:
        return "unknown"
    values = df[id_col].astype(str)
    if values.str.startswith("INC").any():
        return "incident"
    if values.str.startswith("SCTASK").any():
        return "request"
    if values.str.startswith("IMS").any():
        return "interaction"
    if values.str.startswith("TASK").any():
        return "incident_task"
    if values.str.startswith("AINST").any():
        return "survey"
    return "unknown"


def standardize_columns(df, dataset_type):
    rules = STANDARDIZATION_RULES.get(dataset_type, {})
    rename_map = {col: next((v for k, v in rules.items(
    ) if k in col.lower()), col.lower()) for col in df.columns}
    return df.rename(columns=rename_map)


def map_group_column(df):
    if "assignment_group" in df.columns:
        df["group"] = df["assignment_group"].map(group_mapping).fillna("Other")
    else:
        df["group"] = "Unknown"
    return df


# Function for trend delta and pill colors
def get_trend(current, previous):
    delta = current - previous
    delta_pct = (delta / previous * 100) if previous != 0 else 0
    if delta > 0:
        arrow = "↑"
        pill_bg = "#e8f9ee"
        color = "#158237"
    elif delta < 0:
        arrow = "↓"
        pill_bg = "#ffe9e9"
        color = "#c45255"
    else:
        arrow = "→"
        pill_bg = "lightgray"
        color = "black"
    return delta, delta_pct, arrow, pill_bg, color


# ---------------- TAB NAVIGATION ----------------
tabs = ["Data", "Reports"]  # Removed the "Schedule" tab
selected = option_menu(
    menu_title=None,
    options=tabs,
    icons=["database", "file-earmark-bar-graph"],
    default_index=tabs.index(st.session_state.selected_tab),
    orientation="horizontal",
    key="main_nav"
)

if selected != st.session_state.selected_tab:
    st.session_state.selected_tab = selected

page = st.session_state.selected_tab

# ===================== DATA PAGE =====================
if page == "Data":
    st.title("ReportNow")
    st.write("Upload dataset(s) or log in with ServiceNow to generate reports.")

    option = st.radio(
        "Data source",
        ("Upload one or more datasets (CSV or Excel)",
         "Log in with ServiceNow (Coming Soon)"),
        label_visibility="collapsed"
    )

    if option == "Upload one or more datasets (CSV or Excel)":
        uploaded_files = st.file_uploader(
            "Upload dataset(s)", type=["csv", "xlsx"], accept_multiple_files=True
        )

        if uploaded_files:
            for file in uploaded_files:
                if file.name not in st.session_state.datasets:
                    df = pd.read_csv(
                        file, encoding="ISO-8859-1") if file.name.endswith(".csv") else pd.read_excel(file)
                    df.columns = [c.strip() for c in df.columns]
                    dtype = detect_dataset_type(df)
                    df = standardize_columns(df, dtype)
                    df = map_group_column(df)
                    st.session_state.datasets[file.name] = {
                        "data": df, "type": dtype}

    if st.session_state.datasets:
        selected_name = st.selectbox(
            "**Preview standardized dataset**", list(st.session_state.datasets.keys()))
        dataset = st.session_state.datasets[selected_name]
        df = dataset["data"]

        st.write(
            f"**Type:** {dataset['type']} | **Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")
        st.dataframe(df.head())

        st.write("**Edit Groups (double click to edit)**")
        group_table = df[['assignment_group', 'group']
                         ].drop_duplicates().reset_index(drop=True)
        edited_group_table = st.data_editor(
            group_table, num_rows="dynamic", use_container_width=True)

        for _, row in edited_group_table.iterrows():
            df.loc[df['assignment_group'] ==
                   row['assignment_group'], 'group'] = row['group']

        st.session_state.datasets[selected_name]["data"] = df

        if st.button("Generate Report"):
            st.session_state.selected_tab = "Reports"

    elif option == "Log in with ServiceNow (Coming Soon)":
        st.info("ServiceNow OAuth login is planned but currently disabled.\nThis demo uses uploaded datasets to validate the analytics experience.")
        st.button("Login with ServiceNow", disabled=True)


# ===================== REPORTS PAGE =====================
elif page == "Reports":

    today = datetime.today()
    today_str = today.strftime("%B %d, %Y")
    st.title(f"{today_str} Report")

    if not st.session_state.datasets:
        st.warning(
            "No dataset uploaded yet. Report feature will be available after uploading data."
        )
    else:
        # Select the dataset
        selected_name = st.selectbox(
            "**Select dataset**", list(st.session_state.datasets.keys())
        )
        dataset = st.session_state.datasets[selected_name]
        df = dataset["data"]

        # =====================
        # Machine Learning Code (Custom Functions)
        # =====================

        # Step 1: Convert date columns to datetime if not already
        df["opened"] = pd.to_datetime(df["opened"], errors="coerce")
        df["closed"] = pd.to_datetime(df["closed"], errors="coerce")

        # Step 2: Build the weekly timeline of backlog
        def build_assignment_group_weekly_timeline(
            df,
            opened_col="opened",
            closed_col="closed",
            group_col="group",
            assignment_col="assignment_group"
        ):
            df = df.copy()

            df[group_col] = df[group_col].fillna("Unknown").astype(str)
            df[assignment_col] = df[assignment_col].fillna(
                "Unknown").astype(str)

            def agg(col):
                return (
                    df.dropna(subset=[col])
                      .assign(week=lambda x: x[col].dt.to_period("W").dt.start_time)
                      .groupby([group_col, assignment_col, 'week'], as_index=False)
                      .size()
                )

            opened = agg(opened_col).rename(columns={'size': 'opened'})
            closed = agg(closed_col).rename(columns={'size': 'closed'})

            timeline = (
                pd.merge(opened, closed,
                         on=[group_col, assignment_col, 'week'],
                         how='outer')
                .fillna(0)
            )

            timeline[['opened', 'closed']] = timeline[[
                'opened', 'closed']].astype(int)

            return timeline.sort_values([group_col, assignment_col, 'week']).reset_index(drop=True)

        # Generate the timeline data
        timeline = build_assignment_group_weekly_timeline(df)

        # Step 3: Compute backlog over time
        def compute_backlog(
            timeline,
            opened_col="opened",
            closed_col="closed",
            time_col="week",
            current_backlog=0
        ):
            timeline = timeline.sort_values(time_col).copy()
            timeline["backlog"] = 0
            timeline.iloc[-1,
                          timeline.columns.get_loc("backlog")] = current_backlog

            for i in range(len(timeline) - 2, -1, -1):
                timeline.iloc[i, timeline.columns.get_loc("backlog")] = (
                    timeline.iloc[i + 1]["backlog"]
                    - timeline.iloc[i + 1][opened_col]
                    + timeline.iloc[i + 1][closed_col]
                )

            timeline["backlog"] = timeline["backlog"].clip(lower=0).astype(int)

            return timeline

        def compute_backlog_assignment_groups(
            timeline,
            current_backlog_dict,
            opened_col="opened",
            closed_col="closed",
            time_col="week"
        ):
            result = []

            for (group, assignment_group), df_grp in timeline.groupby(
                ['group', 'assignment_group'], dropna=False
            ):
                key = (group, assignment_group)
                current_backlog = current_backlog_dict.get(key, 0)

                df_backlog = compute_backlog(
                    df_grp,
                    opened_col=opened_col,
                    closed_col=closed_col,
                    time_col=time_col,
                    current_backlog=current_backlog
                )

                df_backlog['group'] = group
                df_backlog['assignment_group'] = assignment_group

                result.append(df_backlog)

            return pd.concat(result, ignore_index=True)

        # Step 4: Current backlog per group
        current_backlog_per_group = (
            df[df['active'] == True]
            .groupby(['group', 'assignment_group'])
            .size()
            .reset_index(name='current_backlog')
        )

        current_backlog_dict = {
            (row['group'], row['assignment_group']): row['current_backlog']
            for _, row in current_backlog_per_group.iterrows()
        }

        total_current_backlog = int(df['active'].sum())

        # Step 5: Compute the backlog assignment groups
        timeline = compute_backlog_assignment_groups(
            timeline=timeline,
            current_backlog_dict=current_backlog_dict
        )

        def compute_holidays(df, date_cols=['opened', 'closed']):

            min_year = int(min(df[col].dropna().dt.year.min()
                           for col in date_cols))
            max_year = int(max(df[col].dropna().dt.year.max()
                           for col in date_cols)) + 1

            # --- Prepare US Holidays ---
            us_holidays = holidays.US(years=range(
                min_year, max_year), observed=False)
            allowed_holidays = {
                "New Year's Day", "Martin Luther King Jr. Day", "Memorial Day",
                "Juneteenth", "Independence Day", "Labor Day", "Veterans Day",
                "Thanksgiving Day", "Christmas Day"
            }

            def apply_observed(date):
                if date.weekday() == 5:
                    return date - timedelta(days=1)
                elif date.weekday() == 6:
                    return date + timedelta(days=1)
                return date

            def previous_weekday(d):
                while d.weekday() >= 5:
                    d -= timedelta(days=1)
                return d

            def is_weekday(d):
                return d.weekday() < 5

            # --- Build holiday list ---
            holiday_rows = []
            for date, name in us_holidays.items():
                if name not in allowed_holidays and "Juneteenth" not in name:
                    continue
                if "Juneteenth" in name and date.year < 2023:
                    continue
                observed_date = apply_observed(date)
                holiday_rows.append((name, observed_date))

            # Thanksgiving Closure
            for date, name in us_holidays.items():
                if name == "Thanksgiving Day":
                    observed = apply_observed(date)
                    closure = apply_observed(observed + timedelta(days=1))
                    holiday_rows.append(("Thanksgiving Closure", closure))

            # Christmas Eve
            for date, name in us_holidays.items():
                if name == "Christmas Day":
                    christmas_eve = previous_weekday(date - timedelta(days=1))
                    holiday_rows.append(("Christmas Eve", christmas_eve))

            # Winter Closure (4 weekdays after Christmas)
            for date, name in us_holidays.items():
                if name == "Christmas Day":
                    observed = apply_observed(date)
                    days_added = 0
                    current = observed + timedelta(days=1)
                    while days_added < 4:
                        if is_weekday(current):
                            holiday_rows.append(("Winter Closure", current))
                            days_added += 1
                        current += timedelta(days=1)

            # --- holiday_dates: raw holidays & closures ---
            holiday_df = pd.DataFrame(holiday_rows, columns=[
                                      'holiday', 'date']).drop_duplicates()
            holiday_df['date'] = pd.to_datetime(holiday_df['date'])

            # --- Compute incident counts for analysis ---
            results = holiday_df.copy()
            results['week'] = results['date'].dt.isocalendar().week
            results['year'] = results['date'].dt.isocalendar().year

            for col in date_cols:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                df['_week'] = df[col].dt.isocalendar().week
                df['_year'] = df[col].dt.isocalendar().year

                # Exact incidents on each holiday date
                results[col + '_incidents'] = results['date'].apply(
                    lambda d: (df[col].dt.date == d.date()).sum()
                )

                # Weekly incidents (count incidents in same week/year)
                results[col + '_week_incidents'] = results.apply(
                    lambda row: ((df['_year'] == row['year']) & (
                        df['_week'] == row['week'])).sum(),
                    axis=1
                )

            # Combine multi-day closures in the same week/year
            combine_map = {
                "Winter Closure": ["Winter Closure", "Christmas Day", "Christmas Eve", "New Year's Day"],
                "Thanksgiving Closure": ["Thanksgiving Day", "Thanksgiving Closure"]
            }

            def get_combined_name(name):
                for combined_name, holidays_list in combine_map.items():
                    if name in holidays_list:
                        return combined_name
                return name

            results['holiday_combined'] = results['holiday'].apply(
                get_combined_name)

            holiday_counts_week = results.groupby(['year', 'week', 'holiday_combined'], as_index=False).agg({
                'date': 'min',
                **{col+'_incidents': 'sum' for col in date_cols},
                # weekly incidents
                **{col+'_week_incidents': 'max' for col in date_cols}
            }).sort_values(['year', 'week']).reset_index(drop=True)

            df.drop(columns=['_year', '_week'], inplace=True, errors='ignore')

            # --- RETURN BOTH ---
            return holiday_df, holiday_counts_week

        # Compute holidays
        holiday_df, holiday_counts_week = compute_holidays(
            df, date_cols=['opened', 'closed'])

        def engineer_features(
            timeline,
            holiday_df,
            group_cols=None
        ):

            timeline = timeline.copy()
            holiday_df = holiday_df.copy()

            timeline['week'] = pd.to_datetime(timeline['week'])
            holiday_df['date'] = pd.to_datetime(holiday_df['date'])

            if group_cols is None:
                group_cols = []

            # Encode group columns
            for col in group_cols:
                timeline[f'{col}_code'] = timeline[col].astype(
                    'category').cat.codes

            timeline['week_start'] = timeline['week'] - pd.to_timedelta(
                timeline['week'].dt.weekday, unit='d'
            )
            holiday_df['week_start'] = holiday_df['date'] - pd.to_timedelta(
                holiday_df['date'].dt.weekday, unit='d'
            )

            # Holiday and closure weeks
            closure_weeks = (
                holiday_df[holiday_df['holiday'].isin(
                    ['Winter Closure', 'Thanksgiving Closure'])]
                .drop_duplicates('week_start')
                .assign(is_closure_week=1)
                [['week_start', 'is_closure_week']]
            )

            holiday_weeks = (
                holiday_df[['week_start']]
                .drop_duplicates()
                .assign(is_holiday_week=1)
            )

            timeline = timeline.merge(
                holiday_weeks, on='week_start', how='left')
            timeline = timeline.merge(
                closure_weeks, on='week_start', how='left')

            timeline[['is_holiday_week', 'is_closure_week']] = (
                timeline[['is_holiday_week', 'is_closure_week']]
                .fillna(0)
                .astype(int)
            )

            timeline['holiday_prev_week'] = timeline['is_holiday_week'].shift(
                -1, fill_value=0)
            timeline['holiday_next_week'] = timeline['is_holiday_week'].shift(
                1, fill_value=0)

            # Temporal features
            timeline['month'] = timeline['week'].dt.month
            timeline['weekofyear'] = timeline['week'].dt.isocalendar(
            ).week.astype(int)
            timeline['is_weekend'] = (
                timeline['week'].dt.weekday >= 5).astype(int)
            timeline['is_end_of_month'] = timeline['week'].dt.is_month_end.astype(
                int)

            # Cyclical encoding
            timeline['month_sin'] = np.sin(2*np.pi*timeline['month']/12)
            timeline['month_cos'] = np.cos(2*np.pi*timeline['month']/12)
            timeline['weekday_sin'] = np.sin(
                2*np.pi*timeline['week'].dt.weekday/7)
            timeline['weekday_cos'] = np.cos(
                2*np.pi*timeline['week'].dt.weekday/7)

            # Semester features
            timeline['is_fall_semester_start'] = (
                (timeline['week'].dt.month == 8) &
                (timeline['week'].dt.day.between(20, 28)) &
                (timeline['week'].dt.weekday == 0)
            ).astype(int)

            timeline['is_spring_semester_start'] = 0
            for year in timeline['week'].dt.year.unique():
                jan_mondays = timeline[
                    (timeline['week'].dt.year == year) &
                    (timeline['week'].dt.month == 1) &
                    (timeline['week'].dt.weekday == 0)
                ].sort_values('week')
                if len(jan_mondays) >= 2:
                    timeline.loc[jan_mondays.index[1],
                                 'is_spring_semester_start'] = 1

            timeline['is_semester_start_window'] = (
                timeline[['is_fall_semester_start', 'is_spring_semester_start']]
                .rolling(3, center=True, min_periods=1)
                .max()
                .max(axis=1)
            ).astype(int)

            # =========================
            # --- GROUP × SEMESTER and HOLIDAY/CLOSURE INTERACTIONS ---
            # =========================
            for grp in group_cols:
                # Semester interactions
                for sem_col in ['is_fall_semester_start', 'is_spring_semester_start', 'is_semester_start_window']:
                    timeline[f'{grp}_{sem_col}'] = (
                        (timeline[grp] == timeline[grp]) & (timeline[sem_col] == 1)).astype(int)

                # Holiday/Closure interactions
                for event_col in ['is_holiday_week', 'is_closure_week']:
                    timeline[f'{grp}_{event_col}'] = (
                        (timeline[grp] == timeline[grp]) & (timeline[event_col] == 1)).astype(int)

            # =========================
            # --- Lag / rolling / EWMA features ---
            # =========================
            EPS = 1e-6
            output = []

            grouped = (
                timeline.groupby(group_cols, group_keys=False)
                if group_cols else [((), timeline)]
            )

            for _, df_grp in grouped:
                df = df_grp.sort_values('week').copy()

                for col in ['opened', 'closed', 'backlog']:
                    for lag in [1, 2, 3, 7, 14, 28]:
                        df[f'{col}_lag_{lag}'] = df[col].shift(lag)

                    for window in [3, 4, 7, 14]:
                        df[f'{col}_roll_{window}'] = df[col].shift(
                            1).rolling(window).mean()

                    for span in [3, 7, 14, 28]:
                        df[f'{col}_ewma_{span}'] = df[col].shift(
                            1).ewm(span=span, adjust=False).mean()

                for window in [4, 8]:
                    df[f'opened_closed_ratio_{window}w'] = (
                        df['opened'].shift(1).rolling(window).sum()
                        / (df['closed'].shift(1).rolling(window).sum() + EPS)
                    )
                    df[f'opened_minus_closed_{window}w'] = (
                        df['opened'].shift(1).rolling(window).sum()
                        - df['closed'].shift(1).rolling(window).sum()
                    )

                df['backlog_diff_1'] = df['backlog'] - df['backlog'].shift(1)
                df['backlog_diff_4'] = df['backlog'] - df['backlog'].shift(4)
                df['backlog_roll_4'] = df['backlog'].shift(1).rolling(4).mean()
                df['backlog_roll_8'] = df['backlog'].shift(1).rolling(8).mean()
                df['backlog_trend_4'] = df['backlog'] - df['backlog_roll_4']
                df['opened_to_backlog_ratio'] = df['opened'] / \
                    (df['backlog'] + 1)
                df['closed_to_backlog_ratio'] = df['closed'] / \
                    (df['backlog'] + 1)
                df['holiday_count_last_4w'] = df['is_holiday_week'].shift(
                    1).rolling(4).sum()
                df['closure_count_last_4w'] = df['is_closure_week'].shift(
                    1).rolling(4).sum()

                output.append(df)

            timeline = pd.concat(output, ignore_index=True)

            # Drop rows with NaNs caused by lagging
            temporal_cols = [c for c in timeline.columns if any(
                k in c for k in ['lag', 'roll', 'ewma', 'ratio', 'diff', 'trend'])]
            timeline = timeline.dropna(
                subset=temporal_cols).reset_index(drop=True)

            # Feature sets
            GROUP_FEATURES = [f'{c}_code' for c in group_cols]

            BASE_FEATURES = [
                'month', 'weekofyear', 'is_weekend', 'is_end_of_month',
                'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos',
                'is_holiday_week', 'is_closure_week',
                'holiday_prev_week', 'holiday_next_week',
                'holiday_count_last_4w', 'closure_count_last_4w',
                'is_fall_semester_start', 'is_spring_semester_start',
                'is_semester_start_window',
                'backlog_diff_1', 'backlog_diff_4',
                'backlog_roll_4', 'backlog_roll_8', 'backlog_trend_4',
                'opened_to_backlog_ratio', 'closed_to_backlog_ratio'
            ] + GROUP_FEATURES

            FEATURE_SETS = {
                'opened': [c for c in timeline.columns if c.startswith('opened_') or c in BASE_FEATURES],
                'closed': [c for c in timeline.columns if c.startswith('closed_') or c in BASE_FEATURES]
            }

            timeline.drop(columns=['week_start'], inplace=True)

            return timeline, FEATURE_SETS

        timeline, FEATURE_SETS = engineer_features(
            timeline=timeline,
            holiday_df=holiday_df,
            group_cols=['group', 'assignment_group']
        )

        def time_based_split(df, test_weeks=26, return_split_date=False):
            split_date = df['week'].max() - pd.Timedelta(weeks=test_weeks)
            train_df = df[df['week'] < split_date].copy()
            test_df = df[df['week'] >= split_date].copy()

            if return_split_date:
                return train_df, test_df, split_date
            return train_df, test_df

        train_df, test_df = time_based_split(timeline, test_weeks=26)

        def select_best_num_features(target_col, FEATURE_SETS, train_df,
                                     min_features=5, max_features=20, step=1,
                                     patience=3, n_splits=5, return_ranked=False):

            features = FEATURE_SETS[target_col]
            X = train_df[features]
            y = train_df[target_col]

            # Fit initial model to rank feature importance
            model = GradientBoostingRegressor(
                random_state=42, n_estimators=200, max_depth=3)
            model.fit(X, y)
            importance_scores = pd.Series(
                model.feature_importances_, index=features)
            ranked_features = importance_scores.sort_values(
                ascending=False).index.tolist()

            best_mae, best_features, no_improve = float('inf'), None, 0
            tscv = TimeSeriesSplit(n_splits=n_splits)

            for n in range(min_features, min(max_features, len(ranked_features)) + 1, step):
                top_features = ranked_features[:n]
                fold_maes = []

                for train_idx, val_idx in tscv.split(X):
                    m = clone(model)
                    m.fit(X.iloc[train_idx][top_features], y.iloc[train_idx])
                    preds = m.predict(X.iloc[val_idx][top_features])
                    fold_maes.append(mean_absolute_error(
                        y.iloc[val_idx], preds))

                mean_mae = np.mean(fold_maes)
                print(f"Top {n} features → CV MAE: {mean_mae:.3f}")

                if mean_mae < best_mae:
                    best_mae, best_features, no_improve = mean_mae, top_features, 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"Early stopping at {n} features")
                        break

            print(f"\n✅ Best features ({len(best_features)}): {best_features}")
            if return_ranked:
                return best_features, ranked_features
            return best_features

        def compute_metrics(model_name, model, X_train, y_train, X_test, y_test, tscv=None):
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            rmsle = np.sqrt(
                np.mean((np.log1p(y_test) - np.log1p(np.clip(y_pred, 0, None)))**2))
            poisson = mean_poisson_deviance(
                y_test, np.clip(y_pred, 1e-6, None))

            cv_mae_mean, cv_mae_std, cv_rmsle_mean = None, None, None
            if tscv is not None:
                cv_mae, cv_rmsle = [], []
                for train_idx, val_idx in tscv.split(X_train):
                    model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
                    val_pred = model.predict(X_train.iloc[val_idx])
                    val_true = y_train.iloc[val_idx]
                    cv_mae.append(mean_absolute_error(val_true, val_pred))
                    cv_rmsle.append(np.sqrt(
                        np.mean((np.log1p(val_true) - np.log1p(np.clip(val_pred, 0, None)))**2)))
                cv_mae_mean = np.mean(cv_mae)
                cv_mae_std = np.std(cv_mae)
                cv_rmsle_mean = np.mean(cv_rmsle)

            return {
                'model': model_name,
                'MAE': mae,
                'RMSE': rmse,
                'RMSLE': rmsle,
                'Poisson_deviance': poisson,
                'CV_MAE': cv_mae_mean,
                'CV_MAE_STD': cv_mae_std,
                'CV_RMSLE': cv_rmsle_mean
            }

        def naive_metrics(y_train, y_test):
            naive_pred = y_test.shift(1).iloc[1:]
            naive_actual = y_test.iloc[1:]
            mae = mean_absolute_error(naive_actual, naive_pred)
            rmse = np.sqrt(mean_squared_error(naive_actual, naive_pred))
            rmsle = np.sqrt(
                np.mean((np.log1p(naive_actual) - np.log1p(np.clip(naive_pred, 0, None)))**2))
            poisson = mean_poisson_deviance(
                naive_actual, np.clip(naive_pred, 1e-6, None))
            return {
                'model': 'NAIVE',
                'MAE': mae,
                'RMSE': rmse,
                'RMSLE': rmsle,
                'Poisson_deviance': poisson,
                'CV_MAE': mae,
                'CV_MAE_STD': 0.0,
                'CV_RMSLE': rmsle
            }

        def tune_and_evaluate_model(target_col, train_df, test_df, FEATURE_SETS,
                                    min_features=5, max_features=20,
                                    group_col='group', assignment_col='assignment_group'):

            start_total = time.time()

            # --- Feature selection ---
            selected_features = select_best_num_features(
                target_col, FEATURE_SETS, train_df,
                min_features=min_features, max_features=max_features
            )

            tscv = TimeSeriesSplit(n_splits=5)

            # --- Hyperparameter grids ---
            rf_param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [
                10, 15, 20], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 3], 'max_features': ['sqrt']}
            gb_param_grid = {'n_estimators': [100, 200, 500], 'max_depth': [3, 5, 7], 'learning_rate': [
                0.05, 0.1], 'subsample': [0.8, 1.0], 'min_samples_split': [2, 5]}
            xgb_param_grid = {'n_estimators': [100, 200, 500], 'max_depth': [3, 5, 7], 'learning_rate': [
                0.05, 0.1], 'subsample': [0.8, 1.0], 'colsample_bytree': [0.8, 1.0]}

            # --- Train models ---
            model_searches = {}
            for name, model_class, param_grid in [('RF', RandomForestRegressor, rf_param_grid),
                                                  ('GB', GradientBoostingRegressor,
                                                   gb_param_grid),
                                                  ('XGB', xgb.XGBRegressor, xgb_param_grid)]:
                start_train = time.time()
                search = RandomizedSearchCV(
                    model_class(random_state=42),
                    param_grid,
                    n_iter=10,
                    cv=tscv,
                    scoring='neg_mean_absolute_error',
                    n_jobs=-1,
                    random_state=42
                )
                search.fit(train_df[selected_features], train_df[target_col])
                search.fit_time = time.time() - start_train
                print(f"⏱️ {name} training time: {search.fit_time:.2f}s")
                model_searches[name] = search

            # --- Evaluate models ---
            metrics = []
            for name, search in model_searches.items():
                metric_dict = compute_metrics(name, search.best_estimator_,
                                              train_df[selected_features], train_df[target_col],
                                              test_df[selected_features], test_df[target_col],
                                              tscv)
                metric_dict['train_time_s'] = search.fit_time
                metrics.append(metric_dict)

            # --- Naive baseline ---
            metrics.append(naive_metrics(
                train_df[target_col], test_df[target_col]))

            metrics_df = pd.DataFrame(metrics)
            metrics_df['MAE_rank'] = metrics_df['MAE'].rank()
            metrics_df['RMSE_rank'] = metrics_df['RMSE'].rank()
            metrics_df['RMSLE_rank'] = metrics_df['RMSLE'].rank()
            metrics_df['Poisson_rank'] = metrics_df['Poisson_deviance'].rank()
            metrics_df['score'] = metrics_df['MAE_rank']*0.3 + metrics_df['RMSE_rank'] * \
                0.2 + metrics_df['RMSLE_rank']*0.3 + \
                metrics_df['Poisson_rank']*0.2
            metrics_df['model'] = metrics_df['model'].apply(
                lambda x: f"{x} (Baseline)" if x == 'NAIVE' else x)

            best_row = metrics_df.loc[metrics_df['score'].idxmin()]
            best_model_name = best_row['model']
            best_model = model_searches.get(
                best_model_name.replace(' (Baseline)', ''), None)
            if best_model is not None:
                best_model = best_model.best_estimator_

            total_elapsed = time.time() - start_total
            print(f"\n⏱️ Total elapsed time: {total_elapsed:.2f}s")

            return {
                'best_model_name': best_model_name,
                'best_model': best_model,
                'metrics': metrics_df,
                'models': {k: v.best_estimator_ for k, v in model_searches.items()},
                'selected_features': selected_features
            }

        opened_results = tune_and_evaluate_model(
            target_col='opened',
            train_df=train_df,
            test_df=test_df,
            FEATURE_SETS=FEATURE_SETS
        )

        best_model_opened = opened_results['best_model']
        best_features_opened = opened_results['selected_features']

        closed_results = tune_and_evaluate_model(
            target_col='closed',
            train_df=train_df,
            test_df=test_df,
            FEATURE_SETS=FEATURE_SETS
        )

        best_model_closed = closed_results['best_model']
        best_features_closed = closed_results['selected_features']

        def plot_actual_vs_predicted(df, target_col, model, features, group_col='group', assignment_col='assignment_group', last_n_weeks=26):
            df_plot = df.copy()

            # --- Validate features ---
            missing_cols = [c for c in features if c not in df_plot.columns]
            if missing_cols:
                raise ValueError(f"Missing feature columns: {missing_cols}")

            # --- Global last N weeks ---
            max_week = df_plot['week'].max()
            min_week = max_week - pd.Timedelta(weeks=last_n_weeks-1)
            all_weeks = pd.date_range(
                start=min_week, end=max_week, freq='W-MON')

            # --- Predict for all rows ---
            df_plot['predicted'] = model.predict(df_plot[features])
            df_plot['predicted'] = np.clip(
                df_plot['predicted'], 0, None).round().astype(int)

            # --- Initialize figure ---
            fig = go.Figure()
            valid_combinations, trace_idx = [], 0

            # --- Add traces per group/assignment ---
            for grp in df_plot[group_col].unique():
                for assign in df_plot[assignment_col].unique():
                    sub = df_plot[(df_plot[group_col] == grp) & (
                        df_plot[assignment_col] == assign)]

                    # Reindex to ensure all weeks are included
                    sub = sub.set_index('week').reindex(
                        all_weeks).reset_index().rename(columns={'index': 'week'})

                    if sub.empty or sub[target_col].isna().all():
                        continue

                    # Actual
                    fig.add_trace(go.Scatter(
                        x=sub['week'],
                        y=sub[target_col].fillna(0).astype(int).tolist(),
                        mode='lines+markers',
                        name=f'Actual ({grp}/{assign})',
                        visible=False
                    ))

                    # Predicted
                    fig.add_trace(go.Scatter(
                        x=sub['week'],
                        y=sub['predicted'].fillna(0).tolist(),
                        mode='lines+markers',
                        name=f'Predicted ({grp}/{assign})',
                        visible=False
                    ))

                    valid_combinations.append((grp, assign, trace_idx))
                    trace_idx += 2

            # --- Aggregate all groups ---
            all_actual = df_plot.groupby('week')[target_col].sum()
            all_pred = df_plot.groupby('week')['predicted'].sum()

            all_actual = all_actual.reindex(all_weeks, fill_value=0)
            all_pred = all_pred.reindex(all_weeks, fill_value=0)

            mae_all = mean_absolute_error(all_actual, all_pred)

            fig.add_trace(go.Scatter(
                x=all_actual.index,
                y=all_actual.values.tolist(),
                mode='lines+markers',
                name='Actual (All/All)',
                visible=True
            ))

            fig.add_trace(go.Scatter(
                x=all_pred.index,
                y=all_pred.values.tolist(),
                mode='lines+markers',
                name='Predicted (All/All)',
                visible=True
            ))

            # --- Dropdown buttons ---
            buttons = []

            # All/All
            visible_all = [False] * len(fig.data)
            visible_all[-2:] = [True, True]
            buttons.append(dict(
                label='All/All',
                method='update',
                args=[{'visible': visible_all},
                      {'title': f'{target_col.capitalize()} Actual vs Predicted | MAE: {mae_all:.0f}'}]
            ))

            # Per group/assignment
            for grp, assign, idx in valid_combinations:
                visible = [False] * len(fig.data)
                visible[idx:idx + 2] = [True, True]

                sub = df_plot[(df_plot[group_col] == grp) & (
                    df_plot[assignment_col] == assign)]
                sub = sub.set_index('week').reindex(
                    all_weeks).reset_index().rename(columns={'index': 'week'})
                mae = mean_absolute_error(sub[target_col].fillna(
                    0).astype(int), sub['predicted'].fillna(0))

                buttons.append(dict(
                    label=f'{grp}/{assign}',
                    method='update',
                    args=[{'visible': visible},
                          {'title': f'{target_col.capitalize()} Actual vs Predicted ({grp}/{assign}) | MAE: {mae:.0f}'}]
                ))

            fig.update_layout(
                updatemenus=[dict(active=0, buttons=buttons)],
                title=f'{target_col.capitalize()} Actual vs Predicted',
                xaxis_title='Week',
                yaxis_title=target_col.capitalize(),
                template='plotly_white'
            )

            st.plotly_chart(fig)

        plot_actual_vs_predicted(df=timeline, target_col='opened',
                                 model=opened_results['best_model'], features=opened_results['selected_features'])
        plot_actual_vs_predicted(df=timeline, target_col='closed',
                                 model=closed_results['best_model'], features=closed_results['selected_features'])
