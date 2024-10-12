import pandas as pd
import numpy as np
import datetime
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import sys
from matplotlib import font_manager, rc

font_path = "./assets/NanumSquareRoundOTFR.otf"
font_prop = font_manager.FontProperties(fname=font_path)
rc("font", family=font_prop.get_name())


def get_float_input(prompt, min_value=None, max_value=None):
    while True:
        try:
            value = float(input(prompt))
            if min_value is not None and value < min_value:
                print(f"입력값은 {min_value} 이상이어야 합니다.")
                continue
            if max_value is not None and value > max_value:
                print(f"입력값은 {max_value} 이하이어야 합니다.")
                continue
            return value
        except ValueError:
            print("유효한 숫자를 입력하세요.")


def get_date_input(prompt):
    while True:
        try:
            date_input = input(prompt)
            date = datetime.datetime.strptime(date_input, "%Y %m")
            return date
        except ValueError:
            print("유효한 날짜 형식이 아닙니다. 예: 1972 2")


def check_excel_file(file_path, executable_name):
    if not os.path.exists(file_path):
        print(f"'{file_path}' 파일이 존재하지 않습니다.")
        print(f"'{executable_name}'를 먼저 실행하여 데이터를 먼저 다운로드해 주세요.")
        sys.exit(1)


def main():
    excel_file = "./output/nasdaq_daily_returns.xlsx"
    data_collection_py = "./assets/nasdaq_data_collection.py"
    check_excel_file(excel_file, data_collection_py)

    try:
        nasdaq_data = pd.read_excel(excel_file, parse_dates=["날짜"])
    except Exception as e:
        print(f"엑셀 파일을 읽는 중 오류가 발생했습니다: {e}")
        sys.exit(1)

    nasdaq_data.set_index("날짜", inplace=True)
    nasdaq_data["나스닥 지수"] = nasdaq_data["나스닥 지수"].astype(float)
    nasdaq_data["변동비"] = nasdaq_data["변동비"].astype(float)

    print("===== 나스닥 적립식 매수 시뮬레이션 =====")

    leverage_ratio_1 = get_float_input(
        "차트를 확인할 첫 번째 레버리지 비율을 입력하세요 (예: 2.25): "
    )
    leverage_ratio_2 = get_float_input(
        "차트를 확인할 두 번째 레버리지 비율을 입력하세요 (예: -3): "
    )
    start_month = get_date_input(
        "적립식 매수를 시작할 연과 월을 입력하세요 (예: 1972 2): "
    )
    end_month = get_date_input(
        "적립실 매수를 종료할 연과 월을 입력하세요 (예: 2024 9): "
    ) + pd.offsets.MonthEnd(0)
    monthly_investment = (
        get_float_input(
            "매월 적립식 매수할 금액을 입력하세요 (단위: 만원): ", min_value=0
        )
        * 10000
    )

    investment_data = nasdaq_data[
        (nasdaq_data.index >= start_month) & (nasdaq_data.index <= end_month)
    ].copy()

    investment_data["일간수익률"] = investment_data["변동비"].fillna(0)
    investment_data[f"레버리지_{leverage_ratio_1}_일간수익률"] = (
        investment_data["변동비"] * leverage_ratio_1
    )
    investment_data[f"나스닥 {leverage_ratio_1}배 지수"] = (
        1 + investment_data[f"레버리지_{leverage_ratio_1}_일간수익률"]
    ).cumprod() * investment_data["나스닥 지수"].iloc[0]

    investment_data[f"레버리지_{leverage_ratio_2}_일간수익률"] = (
        investment_data["변동비"] * leverage_ratio_2
    )
    investment_data[f"나스닥 {leverage_ratio_2}배 지수"] = (
        1 + investment_data[f"레버리지_{leverage_ratio_2}_일간수익률"]
    ).cumprod() * investment_data["나스닥 지수"].iloc[0]

    monthly_data = investment_data.resample("ME").last()
    monthly_data["나스닥 전월 대비 변동률"] = monthly_data["나스닥 지수"].pct_change()
    monthly_data[f"나스닥 {leverage_ratio_1}배 전월 대비 변동률"] = monthly_data[
        f"나스닥 {leverage_ratio_1}배 지수"
    ].pct_change()
    monthly_data[f"나스닥 {leverage_ratio_2}배 전월 대비 변동률"] = monthly_data[
        f"나스닥 {leverage_ratio_2}배 지수"
    ].pct_change()

    monthly_data["총 투자 금액"] = 0.0
    monthly_data[f"누적 총 자산({leverage_ratio_1}배)"] = 0.0
    monthly_data[f"누적 총 자산({leverage_ratio_2}배)"] = 0.0

    total_invested = 0.0
    total_shares_1 = 0.0
    total_shares_2 = 0.0

    for idx, row in monthly_data.iterrows():
        price_1 = row[f"나스닥 {leverage_ratio_1}배 지수"]
        price_2 = row[f"나스닥 {leverage_ratio_2}배 지수"]
        if not np.isnan(price_1) and not np.isnan(price_2):
            invested_amount = monthly_investment
            shares_bought_1 = invested_amount / price_1
            shares_bought_2 = invested_amount / price_2
            total_invested += invested_amount
            total_shares_1 += shares_bought_1
            total_shares_2 += shares_bought_2
            asset_value_1 = total_shares_1 * price_1
            asset_value_2 = total_shares_2 * price_2
            monthly_data.at[idx, "총 투자 금액"] = total_invested
            monthly_data.at[idx, f"누적 총 자산({leverage_ratio_1}배)"] = asset_value_1
            monthly_data.at[idx, f"누적 총 자산({leverage_ratio_2}배)"] = asset_value_2

    monthly_data[f"누적 수익률({leverage_ratio_1}배)"] = (
        monthly_data[f"누적 총 자산({leverage_ratio_1}배)"]
        / monthly_data["총 투자 금액"]
        - 1
    )
    monthly_data[f"누적 수익률({leverage_ratio_2}배)"] = (
        monthly_data[f"누적 총 자산({leverage_ratio_2}배)"]
        / monthly_data["총 투자 금액"]
        - 1
    )

    output_df = monthly_data.reset_index()
    output_df["날짜"] = output_df["날짜"].dt.date
    output_df = output_df[
        [
            "날짜",
            "나스닥 지수",
            "나스닥 전월 대비 변동률",
            f"나스닥 {leverage_ratio_1}배 지수",
            f"나스닥 {leverage_ratio_1}배 전월 대비 변동률",
            f"나스닥 {leverage_ratio_2}배 지수",
            f"나스닥 {leverage_ratio_2}배 전월 대비 변동률",
            "총 투자 금액",
            f"누적 수익률({leverage_ratio_1}배)",
            f"누적 수익률({leverage_ratio_2}배)",
            f"누적 총 자산({leverage_ratio_1}배)",
            f"누적 총 자산({leverage_ratio_2}배)",
        ]
    ]
    output_df.columns = [
        "날짜",
        "나스닥 지수",
        "나스닥 전월 대비 변동률",
        f"나스닥 {leverage_ratio_1}배 지수",
        f"나스닥 {leverage_ratio_1}배 전월 대비 변동률",
        f"나스닥 {leverage_ratio_2}배 지수",
        f"나스닥 {leverage_ratio_2}배 전월 대비 변동률",
        "총 투자 금액",
        f"누적 수익률({leverage_ratio_1}배)",
        f"누적 수익률({leverage_ratio_2}배)",
        f"누적 총 자산({leverage_ratio_1}배)",
        f"누적 총 자산({leverage_ratio_2}배)",
    ]

    percentage_columns = [
        "나스닥 전월 대비 변동률",
        f"나스닥 {leverage_ratio_1}배 전월 대비 변동률",
        f"나스닥 {leverage_ratio_2}배 전월 대비 변동률",
        f"누적 수익률({leverage_ratio_1}배)",
        f"누적 수익률({leverage_ratio_2}배)",
    ]
    output_df[percentage_columns] = output_df[percentage_columns].round(4)

    number_columns = [
        "나스닥 지수",
        f"나스닥 {leverage_ratio_1}배 지수",
        f"나스닥 {leverage_ratio_2}배 지수",
    ]
    output_df[number_columns] = output_df[number_columns].round(2)

    monetary_columns = [
        "총 투자 금액",
        f"누적 총 자산({leverage_ratio_1}배)",
        f"누적 총 자산({leverage_ratio_2}배)",
    ]
    output_df[monetary_columns] = output_df[monetary_columns].round(0).astype(np.int64)

    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(output_df["날짜"], output_df["나스닥 지수"], label="나스닥 지수")
    ax1.plot(
        output_df["날짜"],
        output_df[f"나스닥 {leverage_ratio_1}배 지수"],
        label=f"레버리지 {leverage_ratio_1}배 지수",
    )
    ax1.plot(
        output_df["날짜"],
        output_df[f"나스닥 {leverage_ratio_2}배 지수"],
        label=f"레버리지 {leverage_ratio_2}배 지수",
    )
    ax1.set_title("나스닥 지수 비교", fontproperties=font_prop, fontsize=16)
    ax1.set_xlabel("날짜", fontproperties=font_prop)
    ax1.set_ylabel("지수 값", fontproperties=font_prop)
    ax1.legend(prop=font_prop)
    ax1.grid(True)
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(output_df["날짜"], output_df["총 투자 금액"], label="총 투자 금액")
    ax2.plot(
        output_df["날짜"],
        output_df[f"누적 총 자산({leverage_ratio_1}배)"],
        label=f"누적 총 자산 ({leverage_ratio_1}배)",
    )
    ax2.plot(
        output_df["날짜"],
        output_df[f"누적 총 자산({leverage_ratio_2}배)"],
        label=f"누적 총 자산 ({leverage_ratio_2}배)",
    )
    ax2.set_title(
        "총 투자 금액과 누적 총 자산 비교", fontproperties=font_prop, fontsize=16
    )
    ax2.set_xlabel("날짜", fontproperties=font_prop)
    ax2.set_ylabel("금액 (원)", fontproperties=font_prop)
    ax2.legend(prop=font_prop)
    ax2.grid(True)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x):,}원"))
    fig2.tight_layout()

    fig1.savefig("./output/nasdaq_index_comparison.png", dpi=300)
    print(
        "나스닥 지수 비교 차트 이미지가 '/output/nasdaq_index_comparison.png'로 저장되었습니다."
    )

    fig2.savefig("./output/investment_vs_asset_value.png", dpi=300)
    print(
        "총 투자 금액과 누적 총 자산 비교 차트 이미지가 '/output/investment_vs_asset_value.png'로 저장되었습니다."
    )

    plt.show()

    output_file = "./output/investment_simulation.xlsx"
    try:
        with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
            output_df.to_excel(writer, sheet_name="투자 결과", index=False)
            workbook = writer.book
            worksheet = writer.sheets["투자 결과"]

            percentage_format = workbook.add_format(
                {"num_format": "+0.00%;-0.00%;0.00%"}
            )
            comma_format = workbook.add_format({"num_format": "#,##0원"})
            date_format = workbook.add_format({"num_format": "yyyy-mm-dd"})

            for col_name in percentage_columns:
                if col_name in output_df.columns:
                    col_idx = output_df.columns.get_loc(col_name)
                    for row_num in range(len(output_df)):
                        cell_value = output_df.iloc[row_num, col_idx]
                        if pd.isna(cell_value) or math.isnan(cell_value):
                            worksheet.write(row_num + 1, col_idx, "")
                        else:
                            worksheet.write(
                                row_num + 1, col_idx, cell_value, percentage_format
                            )

            for col_name in monetary_columns:
                if col_name in output_df.columns:
                    col_idx = output_df.columns.get_loc(col_name)
                    for row_num in range(len(output_df)):
                        cell_value = output_df.iloc[row_num, col_idx]
                        if pd.isna(cell_value) or math.isnan(cell_value):
                            worksheet.write(row_num + 1, col_idx, "")
                        else:
                            worksheet.write(
                                row_num + 1, col_idx, cell_value, comma_format
                            )

            date_col_idx = output_df.columns.get_loc("날짜")
            worksheet.set_column(date_col_idx, date_col_idx, 15, date_format)

            for col_num, col_name in enumerate(output_df.columns):
                max_len = (
                    max(
                        len(str(col_name)),
                        output_df[col_name].astype(str).map(len).max(),
                    )
                    * 1.5
                    + 2
                )
                worksheet.set_column(col_num, col_num, max_len)

            for col_name in percentage_columns:
                if col_name in output_df.columns:
                    col_idx = output_df.columns.get_loc(col_name)
                    start_row = 1
                    end_row = len(output_df)
                    worksheet.conditional_format(
                        start_row,
                        col_idx,
                        end_row,
                        col_idx,
                        {
                            "type": "cell",
                            "criteria": ">",
                            "value": 0,
                            "format": workbook.add_format({"font_color": "red"}),
                        },
                    )
                    worksheet.conditional_format(
                        start_row,
                        col_idx,
                        end_row,
                        col_idx,
                        {
                            "type": "cell",
                            "criteria": "<",
                            "value": 0,
                            "format": workbook.add_format({"font_color": "blue"}),
                        },
                    )

            max_row = len(output_df)

            chart2 = workbook.add_chart({"type": "line"})
            chart2.add_series(
                {
                    "name": "총 투자 금액",
                    "categories": ["투자 결과", 1, 0, max_row, 0],
                    "values": [
                        "투자 결과",
                        1,
                        output_df.columns.get_loc("총 투자 금액"),
                        max_row,
                        output_df.columns.get_loc("총 투자 금액"),
                    ],
                }
            )
            chart2.add_series(
                {
                    "name": f"누적 총 자산({leverage_ratio_1}배)",
                    "categories": ["투자 결과", 1, 0, max_row, 0],
                    "values": [
                        "투자 결과",
                        1,
                        output_df.columns.get_loc(
                            f"누적 총 자산({leverage_ratio_1}배)"
                        ),
                        max_row,
                        output_df.columns.get_loc(
                            f"누적 총 자산({leverage_ratio_1}배)"
                        ),
                    ],
                }
            )
            chart2.add_series(
                {
                    "name": f"누적 총 자산({leverage_ratio_2}배)",
                    "categories": ["투자 결과", 1, 0, max_row, 0],
                    "values": [
                        "투자 결과",
                        1,
                        output_df.columns.get_loc(
                            f"누적 총 자산({leverage_ratio_2}배)"
                        ),
                        max_row,
                        output_df.columns.get_loc(
                            f"누적 총 자산({leverage_ratio_2}배)"
                        ),
                    ],
                }
            )
            chart2.set_title({"name": "총 투자 금액과 누적 총 자산 비교"})
            chart2.set_x_axis({"name": "날짜", "date_axis": True})
            chart2.set_y_axis({"name": "원", "num_format": "#,##0원"})
            chart2.set_size({"width": 720, "height": 480})
            worksheet.insert_chart("N2", chart2)

            chart1 = workbook.add_chart({"type": "line"})
            chart1.add_series(
                {
                    "name": "나스닥 지수",
                    "categories": ["투자 결과", 1, 0, max_row, 0],
                    "values": [
                        "투자 결과",
                        1,
                        output_df.columns.get_loc("나스닥 지수"),
                        max_row,
                        output_df.columns.get_loc("나스닥 지수"),
                    ],
                }
            )
            chart1.add_series(
                {
                    "name": f"나스닥 {leverage_ratio_1}배 지수",
                    "categories": ["투자 결과", 1, 0, max_row, 0],
                    "values": [
                        "투자 결과",
                        1,
                        output_df.columns.get_loc(f"나스닥 {leverage_ratio_1}배 지수"),
                        max_row,
                        output_df.columns.get_loc(f"나스닥 {leverage_ratio_1}배 지수"),
                    ],
                }
            )
            chart1.add_series(
                {
                    "name": f"나스닥 {leverage_ratio_2}배 지수",
                    "categories": ["투자 결과", 1, 0, max_row, 0],
                    "values": [
                        "투자 결과",
                        1,
                        output_df.columns.get_loc(f"나스닥 {leverage_ratio_2}배 지수"),
                        max_row,
                        output_df.columns.get_loc(f"나스닥 {leverage_ratio_2}배 지수"),
                    ],
                }
            )
            chart1.set_title({"name": "나스닥 지수 비교"})
            chart1.set_x_axis({"name": "날짜", "date_axis": True})
            chart1.set_y_axis({"name": "지수 값"})
            chart1.set_size({"width": 720, "height": 480})
            worksheet.insert_chart("N27", chart1)

    except Exception as e:
        print(f"엑셀 파일을 생성하는 중 오류가 발생했습니다: {e}")
        sys.exit(1)

    final_amount_1 = monthly_data[f"누적 총 자산({leverage_ratio_1}배)"].iloc[-1]
    final_return_1 = (final_amount_1 / total_invested - 1) * 100

    final_amount_2 = monthly_data[f"누적 총 자산({leverage_ratio_2}배)"].iloc[-1]
    final_return_2 = (final_amount_2 / total_invested - 1) * 100

    total_invested = monthly_investment * (monthly_data.shape[0] - 1)
    final_cash_amount = total_invested

    final_return_ratio_1 = final_return_1 / 100 + 1
    final_return_ratio_2 = final_return_2 / 100 + 1

    print("=" * 40)
    print(f"- 현금 보유했을 경우 최종 금액: {final_cash_amount:,.0f}원")
    print("=" * 40)
    print(
        f"- {leverage_ratio_1}배 레버리지를 매달 {monthly_investment // 10000:,.0f}만원씩 적립식 매수했을 경우:"
    )
    print(f"  - 최종 금액: {final_amount_1:,.0f}원")
    print(f"  - 최종 수익률: {final_return_1:.2f}% ({final_return_ratio_1:.2f}배)")
    print("=" * 40)
    print(
        f"- {leverage_ratio_2}배 레버리지를 매달 {monthly_investment // 10000:,.0f}만원씩 적립식 매수했을 경우:"
    )
    print(f"  - 최종 금액: {final_amount_2:,.0f}원")
    print(f"  - 최종 수익률: {final_return_2:.2f}% ({final_return_ratio_2:.2f}배)")
    print("=" * 40)
    print("시뮬레이션이 완료되었습니다.")
    print(f"자세한 데이터는 '{output_file}'에서 확인하세요.")


if __name__ == "__main__":
    main()
