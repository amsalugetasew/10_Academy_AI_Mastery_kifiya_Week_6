from django import forms

class SalesDataForm(forms.Form):
    """Form for collecting sales data for prediction."""

    TransactionId = forms.IntegerField(label="TransactionId")
    # DayOfWeek = forms.ChoiceField(
    #     label="Day of week",
    #     choices=[(1, "first day"), (2, "second day"), (3, "third day"),
    #              (4, "fourth day"), (5, "fifth day"), (6, "sixth day"), (7, "seventh day")]
    # )
    # sales = forms.IntegerField(label="Sales")
    SubscriptionId = forms.IntegerField(label="SubscriptionId")
    AccountId = forms.IntegerField(label="AccountId")
    # forms.DateField(label="Date", widget=forms.TextInput(attrs={"placeholder": "YYYY-MM-DD"}))
    CustomerId = forms.IntegerField(label="CustomerId")
    # Open = forms.ChoiceField(
    #     label="Open status",
    #     choices=[(1, "Open"), (0, "Closed")]
    # )
    # ProviderId = forms.IntegerField(label="ProviderId")
    ProviderId = forms.ChoiceField(
        label="ProviderId",
        choices=[(1, 1), (2, 2), (3, 3),(4, 4), (5, 5),(6, 6)]
    )
    ProductId = forms.ChoiceField(
        label="ProductId",
        choices=[(1, 1), (2, 2), (3, 3),(4, 4), (5, 5),(6, 6), (10, 10), (11, 11),(12, 12), (13, 13),(14, 14), (15, 15)]
    )
    ProductCategory = forms.ChoiceField(
        label="ProductCategory",
        choices=[("financial_services", "financial_services"), ("airtime", "airtime"), ("utility_bill", "utility_bill"),("tv", "tv"), ("ticket", "ticket"),("movies", "movies"),("transport","transport"),("other","other")]
    )
    # ProductCategory = forms.IntegerField(label="ProductCategory")
    # ChannelId = forms.IntegerField(label="ChannelId")
    ChannelId = forms.ChoiceField(
        label="ChannelId",
        choices=[(1, 1), (2, 2), (3, 3), (5, 5)]
    )
    Amount = forms.IntegerField(label="Amount")
    TransactionStartTime = forms.DateField(label="TransactionStartTime", widget=forms.TextInput(attrs={"placeholder": "YYYY-MM-DD"}))
    PricingStrategy = forms.IntegerField(label="PricingStrategy")
    FraudResult = forms.IntegerField(label="FraudResult")
    # StateHoliday = forms.ChoiceField(
    #     label="State holiday",
    #     choices=[('0', "No Holiday"), ('a', "a"), ('b', "b"), ('c', "c")]
    # )
    # SchoolHoliday = forms.ChoiceField(
    #     label="School holiday",
    #     choices=[(1, "School Holiday"), (0, "No School Holiday")]
    # )