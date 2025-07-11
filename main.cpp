#include <iostream>
#include <torch/torch.h>

int add(int a, int b) { return a + b; }

int main() {
    int result = add(3, 4);
    std::cout << "Result is: " << result << std::endl;

    return 0;
}
