from textual.app import App, ComposeResult
from textual.widgets import DataTable, Input, Button
from textual.containers import Container
from textual.reactive import reactive

class EditableListApp(App):
    CSS = """
    Container {
        layout: vertical;
        padding: 1;
    }
    DataTable {
        width: 100%;
        height: 70%;
    }
    Input {
        width: 100%;
        margin-top: 1;
    }
    Button {
        width: 100%;
        margin-top: 1;
    }
    """

    selected_row = reactive(None)  # 跟踪选中的行

    def compose(self) -> ComposeResult:
        yield Container(
            DataTable(id="list"),  # 数据表格
            Input(placeholder="Edit first column...", id="editor"),  # 输入框
            Button("Update", variant="primary", id="update-btn")  # 更新按钮
        )

    def on_mount(self) -> None:
        # 初始化表格数据
        table = self.query_one("#list", DataTable)
        table.add_columns("Name", "Age", "Role")  # 定义列名
        table.add_rows([
            ("Alice", 25, "Engineer"),
            ("Bob", 30, "Designer"),
            ("Charlie", 28, "Manager"),
        ])

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """当用户选中某一行时触发"""
        self.selected_row = event.row_key  # 记录选中行的索引
        row_data = event.data_table.get_row(event.row_key)  # 获取行数据
        self.query_one("#editor", Input).value = row_data[0]  # 将第一列显示在输入框

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """点击更新按钮时修改数据"""
        if self.selected_row is not None:
            new_value = self.query_one("#editor", Input).value  # 获取输入的新值
            table = self.query_one("#list", DataTable)
            row = list(table.get_row(self.selected_row))  # 获取原始行数据
            row[0] = new_value  # 修改第一列
            table.update_cell(self.selected_row, tuple(row))  # 更新表格

if __name__ == "__main__":
    EditableListApp().run()