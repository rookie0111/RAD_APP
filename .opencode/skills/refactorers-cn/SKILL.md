---
name: 代码重构器
description: 代码重构、清理和改进。当用户提到重构、清理、改进、简化、提取、重命名、去重、DRY、重写、现代化、技术债、代码异味，或要求在不改变行为的前提下优化代码时使用。
---

# 代码重构器技能

AI 驱动的重构指导，专注于改善代码结构、可读性和可维护性，同时保持行为不变，强调系统化转换、安全变更和可衡量的改进。

## 技能功能

本技能提供专家级重构指导，包括识别改进机会、规划安全转换、系统化执行变更和验证行为保持。将重构最佳实践与实用的、保持行为的变更相结合。

**核心能力：**
- **代码异味检测**：识别表明深层问题的模式
- **重构模式**：提取、重命名、移动、内联转换
- **安全转换**：保持行为的逐步变更
- **复杂度降低**：简化嵌套逻辑、长方法
- **去重**：发现并整合重复代码
- **现代化**：更新到当前语言惯用写法和 API

## 核心原则

### 重构思维
- **行为优先**：永远不改变代码做什么，只改变怎么做
- **小步前进**：一次一个转换，中间测试
- **测试先行**：重构前确保有测试覆盖
- **适可而止**：完美是足够好的敌人
- **清理营地**：你接触的代码应该变得更好

### 重构质量指标
1. **可读性** — 代码是否更容易理解？
2. **可维护性** — 代码是否更容易修改？
3. **可测试性** — 代码是否更容易测试？
4. **简洁性** — 代码是否更简单？
5. **一致性** — 代码是否遵循统一模式？

## 重构工作流

### 1. 评估
```
分析当前状态：
├── 测试覆盖（有测试吗？覆盖率多少？）
├── 代码异味（哪些模式表明问题？）
├── 复杂度指标（圈复杂度、认知复杂度）
├── 依赖关系（还涉及哪些？）
└── 风险等级（关键路径？频繁变更？）
```

### 2. 规划
```
制定重构计划：
├── 目标（我们要改进什么？）
├── 范围（涉及哪些文件/函数？）
├── 顺序（什么变更顺序？）
├── 安全（如何验证行为？）
└── 回滚（出错如何撤销？）
```

### 3. 执行
```
对每个转换：
├── 运行测试 — 确认基线通过
├── 执行变更 — 单一、聚焦的转换
├── 运行测试 — 确认测试仍通过
├── 提交 — 小的、原子的提交
└── 审查 — 是否改进？继续？
```

### 4. 验证
```
确认成功：
├── 测试通过 — 行为不变
├── 指标改善 — 复杂度降低
├── 代码审查 — 团队认同改进
└── 性能 — 无回退
```

## 代码异味目录

### 膨胀者
| 异味 | 症状 | 重构方式 |
|------|------|----------|
| **长方法** | >20行，多个关注点 | 提取方法 |
| **长参数列表** | >3-4个参数 | 参数对象、构建器 |
| **大类** | >300行，多个职责 | 提取类 |
| **原始类型偏执** | 用原始类型表示领域概念 | 值对象 |
| **数据泥团** | 相同字段总是一起出现 | 提取类 |

### 面向对象滥用者
| 异味 | 症状 | 重构方式 |
|------|------|----------|
| **switch语句** | 重复的 switch/case 按类型分发 | 多态、策略模式 |
| **拒绝继承** | 子类不使用继承的成员 | 组合优于继承 |
| **替代类** | 相似类，不同接口 | 提取接口 |
| **特性依恋** | 方法使用其他类的数据 | 移动方法 |
| **不当亲密** | 类之间知道太多 | 移动方法/字段 |

### 变更阻止者
| 异味 | 症状 | 重构方式 |
|------|------|----------|
| **发散式变更** | 一个类因不同原因被修改 | 提取类 |
| **散弹式修改** | 一个变更涉及多个类 | 移动方法/字段 |
| **平行继承** | 两个层次结构中创建子类 | 合并 |

### 可废弃者
| 异味 | 症状 | 重构方式 |
|------|------|----------|
| **死代码** | 不可达或未使用 | 删除 |
| **投机性泛化** | "为了将来"的未使用抽象 | 内联、删除 |
| **重复代码** | 相同逻辑在多处 | 提取方法/类 |
| **懒惰类** | 类做的事太少 | 内联类 |

## 核心重构模式

### 提取方法
```python
# 重构前：长方法，多个关注点
def process_order(order):
    if not order.items:
        raise ValueError("Empty order")
    if not order.customer:
        raise ValueError("No customer")
    subtotal = sum(item.price * item.quantity for item in order.items)
    tax = subtotal * 0.1
    shipping = 5.99 if subtotal < 50 else 0
    total = subtotal + tax + shipping
    email_body = f"Order total: ${total:.2f}"
    send_email(order.customer.email, "Order Confirmation", email_body)
    return total

# 重构后：小的、聚焦的方法
def process_order(order):
    validate_order(order)
    total = calculate_order_total(order)
    send_order_confirmation(order.customer, total)
    return total

def validate_order(order):
    if not order.items:
        raise ValueError("Empty order")
    if not order.customer:
        raise ValueError("No customer")

def calculate_order_total(order):
    subtotal = sum(item.price * item.quantity for item in order.items)
    tax = subtotal * 0.1
    shipping = 5.99 if subtotal < 50 else 0
    return subtotal + tax + shipping

def send_order_confirmation(customer, total):
    email_body = f"Order total: ${total:.2f}"
    send_email(customer.email, "Order Confirmation", email_body)
```

### 用多态替代条件判断
```python
# 重构前：按类型 switch
def calculate_shipping(order):
    if order.shipping_type == "standard":
        return 5.99
    elif order.shipping_type == "express":
        return 15.99
    elif order.shipping_type == "overnight":
        return 29.99
    else:
        raise ValueError(f"Unknown shipping type: {order.shipping_type}")

# 重构后：多态
from abc import ABC, abstractmethod

class ShippingMethod(ABC):
    @abstractmethod
    def calculate_cost(self, order) -> float:
        pass

class StandardShipping(ShippingMethod):
    def calculate_cost(self, order) -> float:
        return 5.99

class ExpressShipping(ShippingMethod):
    def calculate_cost(self, order) -> float:
        return 15.99

class OvernightShipping(ShippingMethod):
    def calculate_cost(self, order) -> float:
        return 29.99

SHIPPING_METHODS = {
    "standard": StandardShipping(),
    "express": ExpressShipping(),
    "overnight": OvernightShipping(),
}

def calculate_shipping(order):
    method = SHIPPING_METHODS.get(order.shipping_type)
    if not method:
        raise ValueError(f"Unknown shipping type: {order.shipping_type}")
    return method.calculate_cost(order)
```

### 引入参数对象
```python
# 重构前：参数过多
def create_user(first_name, last_name, email, phone,
                street, city, state, zip_code, country):
    pass

# 重构后：分组为对象
from dataclasses import dataclass

@dataclass
class PersonalInfo:
    first_name: str
    last_name: str
    email: str
    phone: str

@dataclass
class Address:
    street: str
    city: str
    state: str
    zip_code: str
    country: str

def create_user(personal_info: PersonalInfo, address: Address):
    pass
```

### 替换魔法数字
```python
# 重构前：魔法数字
def calculate_price(quantity, price):
    if quantity > 10:
        return quantity * price * 0.9
    elif quantity > 5:
        return quantity * price * 0.95
    return quantity * price

# 重构后：命名常量
BULK_DISCOUNT_THRESHOLD = 10
BULK_DISCOUNT_RATE = 0.10

VOLUME_DISCOUNT_THRESHOLD = 5
VOLUME_DISCOUNT_RATE = 0.05

def calculate_price(quantity, price):
    if quantity > BULK_DISCOUNT_THRESHOLD:
        return quantity * price * (1 - BULK_DISCOUNT_RATE)
    elif quantity > VOLUME_DISCOUNT_THRESHOLD:
        return quantity * price * (1 - VOLUME_DISCOUNT_RATE)
    return quantity * price
```

### 简化条件逻辑
```python
# 重构前：嵌套条件（箭头代码）
def get_user_discount(user):
    if user is not None:
        if user.is_active:
            if user.membership:
                if user.membership.level == "gold":
                    return 0.20
                elif user.membership.level == "silver":
                    return 0.10
                else:
                    return 0.05
            else:
                return 0
        else:
            return 0
    else:
        return 0

# 重构后：守卫子句 + 提前返回
def get_user_discount(user):
    if user is None:
        return 0
    if not user.is_active:
        return 0
    if not user.membership:
        return 0
    discount_rates = {"gold": 0.20, "silver": 0.10}
    return discount_rates.get(user.membership.level, 0.05)
```

## 去重策略

### 提取公共方法
```python
# 重构前：两处重复的验证
class UserController:
    def create_user(self, data):
        if not data.get("email"):
            raise ValueError("Email required")
        if "@" not in data.get("email", ""):
            raise ValueError("Invalid email")

    def update_user(self, user_id, data):
        if not data.get("email"):
            raise ValueError("Email required")
        if "@" not in data.get("email", ""):
            raise ValueError("Invalid email")

# 重构后：提取为共享辅助方法
class UserController:
    def _validate_email(self, data):
        email = data.get("email", "")
        if not email:
            raise ValueError("Email required")
        if "@" not in email:
            raise ValueError("Invalid email")

    def create_user(self, data):
        self._validate_email(data)

    def update_user(self, user_id, data):
        self._validate_email(data)
```

### 模板方法模式
```python
# 重构前：相似方法略有差异
def export_csv(data):
    lines = ["name,email,phone"]
    for row in data:
        lines.append(f"{row.name},{row.email},{row.phone}")
    return "\n".join(lines)

def export_tsv(data):
    lines = ["name\temail\tphone"]
    for row in data:
        lines.append(f"{row.name}\t{row.email}\t{row.phone}")
    return "\n".join(lines)

# 重构后：模板 + 自定义点
def export_delimited(data, delimiter=","):
    lines = [delimiter.join(["name", "email", "phone"])]
    for row in data:
        lines.append(delimiter.join([row.name, row.email, row.phone]))
    return "\n".join(lines)

def export_csv(data):
    return export_delimited(data, delimiter=",")

def export_tsv(data):
    return export_delimited(data, delimiter="\t")
```

## 现代化模式

### Python 现代化
```python
# 旧：手动字符串格式化
message = "Hello, %s. You have %d messages." % (name, count)
message = "Hello, {}. You have {} messages.".format(name, count)

# 现代：f-string (3.6+)
message = f"Hello, {name}. You have {count} messages."

# 旧：带默认值的字典访问
value = d.get("key", None)
if value is None:
    value = default_value

# 现代：海象运算符 (3.8+) 或 defaultdict
value = default_value if (v := d.get("key")) is None else v

# 旧：注释中的类型检查
def greet(name):  # type: (str) -> str
    return f"Hello, {name}"

# 现代：类型标注
def greet(name: str) -> str:
    return f"Hello, {name}"
```

## 何时使用此技能

**触发短语：**
- "这代码太乱了..."
- "怎么清理这段代码..."
- "这个函数太长了..."
- "这里有重复代码..."
- "帮我重构..."
- "让这段代码更易读..."
- "这段代码看不懂..."
- "能简化一下吗..."
- "技术债太多了..."
- "代码有异味..."

**示例请求：**
1. "这个函数200行，帮我拆分一下"
2. "同样的代码写了三遍"
3. "这些嵌套 if 怎么去掉"
4. "这个类做了太多事"
5. "帮我现代化这段旧代码"
6. "这个函数很难测试"

## 重构准备检查清单

开始重构前：

- [ ] **有测试吗？** 行保持需要测试来验证
- [ ] **测试通过吗？** 从已知良好状态开始
- [ ] **范围明确吗？** 知道改什么、不改什么
- [ ] **时间限制？** 设定时间避免无底洞
- [ ] **提交点？** 知道中断时在哪里停下
- [ ] **团队对齐？** 其他人知道你在重构这个区域

## 安全重构步骤

### IDE 辅助重构（推荐）
```
大多数现代 IDE 可以自动：
├── 重命名（变量、函数、类、文件）
├── 提取（方法、变量、常量、接口）
├── 内联（方法、变量）
├── 移动（函数、类到不同文件）
├── 更改签名（添加/删除/重排参数）
└── 生成（构造器、getter/setter、委托方法）
有 IDE 重构工具时优先使用 — 它们会更新所有引用。
```

### 手动安全重构
```
1. 运行测试 — 确认通过
2. 做最小的可能变更
3. 运行测试 — 确认仍通过
4. 提交，附带描述性消息
5. 重复直到完成

如果变更后测试失败：
→ 立即撤销 (git checkout)
→ 尝试更小的步骤
```

## 与其他技能的协作

- **架构师**：大规模重构遵循架构指导
- **测试员**：重构前确保测试覆盖
- **审查员**：重构后审查确认改进
- **故障排查员**：有时重构会暴露或修复 bug
