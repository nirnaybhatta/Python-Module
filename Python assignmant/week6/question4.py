class ShoppingCart:
    def __init__(self):
        self.items = []

    def add_item(self, name, price):
        self.items.append({"name": name, "price": price})

    def total_price(self):
        return sum(item["price"] for item in self.items)


if __name__ == "__main__":
    cart = ShoppingCart()
    cart.add_item("Pen", 2.5)
    cart.add_item("Notebook", 5.0)
    cart.add_item("Backpack", 30.0)
    cart.add_item("Water Bottle", 8.0)
    cart.add_item("Headphones", 15.0)
    print(cart.total_price())
