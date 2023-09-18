import customtkinter

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("green")

root = customtkinter.CTk()
root.title("CustomTkinter Tabview")

tabview = customtkinter.CTkTabview(root)
tabview.pack(padx=20, pady=20)

tabview.add("tab 1")
tabview.add("tab 2")
tabview.set("tab 2")

button_1 = customtkinter.CTkButton(tabview.tab("tab 1"), text="Tab 1")
button_1.pack(padx=20, pady=20)

root.mainloop()