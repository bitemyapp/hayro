use hayro_syntax::content;
use hayro_syntax::object::Number;
use hayro_syntax::object::Stream;
use hayro_syntax::reader::Reader;
use hayro_syntax::reader::{ReaderContext, ReaderExt};
use std::array;
use std::cell::RefCell;
use std::ops::Rem;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// A type 4 function (postscript function).
#[derive(Debug)]
pub(crate) struct Type4 {
    program: Vec<PostScriptOp>,
    /// The program compiled for the number of inputs it's first evaluated with, if it can be.
    compiled: OnceLock<Option<Compiled>>,
}

impl Type4 {
    /// Create a new type 4 function.
    pub(crate) fn new(stream: &Stream<'_>) -> Option<Self> {
        Some(Self::from_program(parse_procedure(
            &stream.decoded().ok()?,
        )?))
    }

    fn from_program(program: Vec<PostScriptOp>) -> Self {
        Self {
            program,
            compiled: OnceLock::new(),
        }
    }

    /// Evaluate the function with the given input.
    pub(crate) fn eval(&self, input: &[f32], output: &mut [f32]) -> Option<()> {
        let compiled = self
            .compiled
            .get_or_init(|| Compiled::new(&self.program, input.len()));
        match compiled {
            Some(compiled) if compiled.inputs == input.len() => compiled.run(input, output),
            _ => self.interpret(input, output),
        }
    }

    /// Evaluate the function by interpreting the program.
    fn interpret(&self, input: &[f32], output: &mut [f32]) -> Option<()> {
        // Functions are evaluated for every pixel of some shadings and images, so the stack
        // is reused rather than set up (100 entries) and moved for every evaluation.
        thread_local! {
            static STACK: RefCell<InterpreterStack> = RefCell::new(InterpreterStack::new());
        }

        STACK.with_borrow_mut(|stack| {
            self.run_on(input, stack)?;
            let items = stack.items();
            let values = &items[items.len().checked_sub(output.len())?..];

            for (output, value) in output.iter_mut().zip(values) {
                *output = value.as_f32();
            }

            Some(())
        })
    }

    #[cfg(test)]
    fn run(&self, input: &[f32]) -> Option<InterpreterStack> {
        let mut stack = InterpreterStack::new();
        self.run_on(input, &mut stack)?;

        Some(stack)
    }

    fn run_on(&self, input: &[f32], stack: &mut InterpreterStack) -> Option<()> {
        stack.clear();

        for input in input {
            stack.push(Argument::Float(*input))?;
        }

        eval_inner(&self.program, stack)
    }
}

#[derive(Clone, Copy)]
enum Argument {
    Float(f32),
    Bool(bool),
}

impl Default for Argument {
    fn default() -> Self {
        Self::Float(0.0)
    }
}

impl Argument {
    fn as_bool(&self) -> bool {
        match self {
            Self::Float(f) => *f != 0.0,
            Self::Bool(b) => *b,
        }
    }

    fn as_f32(&self) -> f32 {
        match self {
            Self::Float(f) => *f,
            Self::Bool(b) => {
                if *b {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

struct ArgumentsStack<T: Default, const C: usize> {
    stack: [T; C],
    len: usize,
}

impl<T: Default, const C: usize> ArgumentsStack<T, C> {
    fn new() -> Self {
        Self {
            stack: array::from_fn(|_| T::default()),
            len: 0,
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn clear(&mut self) {
        self.len = 0;
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    fn push(&mut self, n: T) -> Option<()> {
        if self.len == C {
            None
        } else {
            self.stack[self.len] = n;
            self.len += 1;

            Some(())
        }
    }

    #[inline]
    fn at(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            None
        } else {
            Some(&self.stack[index])
        }
    }

    #[inline]
    fn last(&self) -> Option<&T> {
        if self.is_empty() {
            None
        } else {
            Some(&self.stack[self.len - 1])
        }
    }

    #[inline]
    fn pop(&mut self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            self.len -= 1;
            Some(std::mem::take(&mut self.stack[self.len]))
        }
    }

    #[inline]
    fn items(&self) -> &[T] {
        &self.stack[..self.len]
    }

    #[inline]
    fn items_mut(&mut self) -> &mut [T] {
        &mut self.stack[..self.len]
    }
}

type InterpreterStack = ArgumentsStack<Argument, 100>;
type ParseStack = ArgumentsStack<Vec<PostScriptOp>, 2>;

fn eval_inner(procedure: &[PostScriptOp], arg_stack: &mut InterpreterStack) -> Option<()> {
    macro_rules! zero {
        ($eval:expr) => {
            arg_stack.push($eval);
        };
    }

    macro_rules! one_f {
        ($eval:expr) => {
            let n1 = arg_stack.pop()?;
            arg_stack.push(Argument::Float($eval(n1.as_f32())));
        };
    }

    macro_rules! two_f {
        ($eval:expr) => {
            let n2 = arg_stack.pop()?;
            let n1 = arg_stack.pop()?;
            arg_stack.push(Argument::Float($eval(n1.as_f32(), n2.as_f32())));
        };
    }

    macro_rules! four {
        ($eval_f:expr, $eval_b:expr) => {
            let n2 = arg_stack.pop()?;
            let n1 = arg_stack.pop()?;

            let res = match (n1, n2) {
                (Argument::Float(f1), Argument::Float(f2)) => Argument::Float($eval_f(f1, f2)),
                (Argument::Float(_), Argument::Bool(f2)) => {
                    Argument::Bool($eval_b(n1.as_bool(), f2))
                }
                (Argument::Bool(f1), Argument::Float(_)) => {
                    Argument::Bool($eval_b(f1, n2.as_bool()))
                }
                (Argument::Bool(f1), Argument::Bool(f2)) => Argument::Bool($eval_b(f1, f2)),
            };

            arg_stack.push(res);
        };
    }

    for op in procedure {
        match op {
            PostScriptOp::Number(n) => arg_stack.push(Argument::Float(*n))?,
            PostScriptOp::Abs => {
                one_f!(|n: f32| n.abs());
            }
            PostScriptOp::Add => {
                two_f!(|n1: f32, n2: f32| n1 + n2);
            }
            PostScriptOp::Atan => {
                two_f!(atan);
            }
            PostScriptOp::Ceiling => {
                one_f!(|n: f32| n.ceil());
            }
            PostScriptOp::Cos => {
                one_f!(cos);
            }
            PostScriptOp::Cvi => {
                one_f!(|n: f32| n.trunc());
            }
            PostScriptOp::Cvr => {
                one_f!(|n: f32| n);
            }
            PostScriptOp::Div => {
                two_f!(|n1: f32, n2: f32| n1 / n2);
            }
            PostScriptOp::Exp => {
                two_f!(|n1: f32, n2: f32| n1.powf(n2));
            }
            PostScriptOp::Floor => {
                one_f!(|n: f32| n.floor());
            }
            PostScriptOp::Idiv => {
                let n2 = arg_stack.pop()?;
                let n1 = arg_stack.pop()?;
                arg_stack.push(Argument::Float(idiv(n1.as_f32(), n2.as_f32())?));
            }
            PostScriptOp::Ln => {
                one_f!(|n: f32| n.ln());
            }
            PostScriptOp::Log => {
                one_f!(|n: f32| n.log10());
            }
            PostScriptOp::Mod => {
                two_f!(|n1: f32, n2: f32| n1.rem(n2));
            }
            PostScriptOp::Mul => {
                two_f!(|n1: f32, n2: f32| n1 * n2);
            }
            PostScriptOp::Neg => {
                one_f!(|n: f32| -n);
            }
            PostScriptOp::Round => {
                one_f!(|n: f32| n.round());
            }
            PostScriptOp::Sin => {
                one_f!(sin);
            }
            PostScriptOp::Sqrt => {
                one_f!(|n: f32| n.sqrt());
            }
            PostScriptOp::Sub => {
                two_f!(|n1: f32, n2: f32| n1 - n2);
            }
            PostScriptOp::Truncate => {
                one_f!(|n: f32| n.trunc());
            }
            PostScriptOp::And => {
                two_f!(and);
            }
            PostScriptOp::Bitshift => {
                two_f!(bitshift);
            }
            PostScriptOp::Eq => {
                two_f!(|n1: f32, n2: f32| bf(n1 == n2));
            }
            PostScriptOp::False => {
                zero!(Argument::Bool(false));
            }
            PostScriptOp::Ge => {
                two_f!(|n1: f32, n2: f32| bf(n1 >= n2));
            }
            PostScriptOp::Gt => {
                two_f!(|n1: f32, n2: f32| bf(n1 > n2));
            }
            PostScriptOp::Le => {
                two_f!(|n1: f32, n2: f32| bf(n1 <= n2));
            }
            PostScriptOp::Lt => {
                two_f!(|n1: f32, n2: f32| bf(n1 < n2));
            }
            PostScriptOp::Ne => {
                two_f!(|n1: f32, n2: f32| bf(n1 != n2));
            }
            PostScriptOp::Not => {
                let arg = arg_stack.pop()?;

                let res = match arg {
                    Argument::Float(f) => Argument::Float(not(f)),
                    Argument::Bool(b) => Argument::Bool(!b),
                };

                arg_stack.push(res);
            }
            PostScriptOp::Or => {
                four!(or, |b1: bool, b2: bool| b1 || b2);
            }
            PostScriptOp::True => {
                zero!(Argument::Bool(true));
            }
            PostScriptOp::Xor => {
                four!(xor, |b1: bool, b2: bool| b1 ^ b2);
            }
            PostScriptOp::If(p) => {
                let cond = arg_stack.pop()?.as_bool();

                if cond {
                    eval_inner(p, arg_stack)?;
                }
            }
            PostScriptOp::IfElse(p1, p2) => {
                let cond = arg_stack.pop()?.as_bool();

                if cond {
                    eval_inner(p1, arg_stack)?;
                } else {
                    eval_inner(p2, arg_stack)?;
                }
            }
            PostScriptOp::Copy => {
                let n = arg_stack.pop()?.as_f32() as u32 as usize;
                let start = arg_stack.len().checked_sub(n)?;
                for i in start..arg_stack.len() {
                    arg_stack.push(*arg_stack.at(i)?);
                }
            }
            PostScriptOp::Dup => {
                arg_stack.push(*arg_stack.last()?);
            }
            PostScriptOp::Exch => {
                let n2 = arg_stack.pop()?;
                let n1 = arg_stack.pop()?;

                arg_stack.push(n2);
                arg_stack.push(n1);
            }
            PostScriptOp::Index => {
                let n = arg_stack.pop()?.as_f32() as u32 as usize;
                let n = arg_stack.len().checked_sub(n + 1)?;

                arg_stack.push(*arg_stack.at(n)?);
            }
            PostScriptOp::Pop => {
                arg_stack.pop()?;
            }
            PostScriptOp::Roll => {
                let j = arg_stack.pop()?.as_f32() as i32;
                let n = arg_stack.pop()?.as_f32() as u32 as usize;
                let trimmed_n = arg_stack.len().checked_sub(n)?;

                roll(&mut arg_stack.items_mut()[trimmed_n..], j);
            }
        }
    }

    Some(())
}

// The operators' results, shared by the interpreter and compiled programs so that both give
// the same values.

fn bf(cond: bool) -> f32 {
    (cond as i32) as f32
}

fn atan(n1: f32, n2: f32) -> f32 {
    let mut res = n1.atan2(n2).to_degrees() % 360.0;
    if res < 0.0 {
        res += 360.0;
    }

    res
}

fn cos(n: f32) -> f32 {
    n.to_radians().cos()
}

fn sin(n: f32) -> f32 {
    n.to_radians().sin()
}

/// `None` for a division by zero (or of the least integer by -1), which PostScript gives an
/// error for.
fn idiv(n1: f32, n2: f32) -> Option<f32> {
    (n1 as i32).checked_div(n2 as i32).map(|q| q as f32)
}

fn and(n1: f32, n2: f32) -> f32 {
    (n1 as i32 & n2 as i32) as f32
}

fn or(n1: f32, n2: f32) -> f32 {
    ((n1 as i32) | (n2 as i32)) as f32
}

fn xor(n1: f32, n2: f32) -> f32 {
    ((n1 as i32) ^ (n2 as i32)) as f32
}

fn not(n: f32) -> f32 {
    !(n as i32) as f32
}

/// Shifts of 32 or more bits wrap, as Rust's shifts do in release builds.
fn bitshift(n1: f32, n2: f32) -> f32 {
    let num = n1 as u32;
    let shift = n2 as i32;

    if shift >= 0 {
        num.wrapping_shl(shift as u32) as f32
    } else {
        num.wrapping_shr(shift.unsigned_abs()) as f32
    }
}

/// Roll the top of the stack, `target`, by `j`: up when it's positive.
fn roll<T>(target: &mut [T], j: i32) {
    if target.is_empty() {
        return;
    }

    if j >= 0 {
        target.rotate_right(j as usize % target.len());
    } else {
        target.rotate_left(j.unsigned_abs() as usize % target.len());
    }
}

/// The registers of a compiled program: enough for most programs; others are interpreted.
const REGISTERS: usize = 256;
/// The interpreter's stack's capacity, which a compiled program fails beyond as well.
const STACK_CAPACITY: usize = 100;

/// A program compiled for a number of inputs, as instructions on registers. Each value the
/// program pushes gets a register, and where the stack holds each one is worked out ahead of
/// time (there are no loops), so operators read and write registers rather than popping and
/// pushing, operators that only rearrange the stack cost nothing, and numbers are in their
/// registers from the start. Where the interpreter would fail (the stack over- or
/// underflowing), the compiled program fails when it gets there. Programs whose stack can't
/// be worked out ahead of time (`copy`, `index` or `roll` of a computed count, branches
/// leaving the stack different, or too many values) are interpreted instead.
#[derive(Debug)]
struct Compiled {
    /// Distinguishes the program from others, for [`REGISTER_FILE`].
    id: u64,
    inputs: usize,
    code: Vec<Instruction>,
    /// The registers as a run starts, with the program's numbers.
    registers: Box<[f32; REGISTERS]>,
    /// The registers of the stack at the end.
    stack: Vec<u8>,
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

thread_local! {
    /// Each thread's registers, with the numbers of the program that last ran on it (by its
    /// id). A program doesn't change those, and sets the others before it reads them, so
    /// that running it again only needs its inputs, rather than a copy of all its registers.
    static REGISTER_FILE: RefCell<(u64, [f32; REGISTERS])> =
        const { RefCell::new((0, [0.0; REGISTERS])) };
}

/// An instruction of a compiled program: a destination register first, then operands.
#[derive(Debug, Clone, Copy)]
enum Instruction {
    Move(u8, u8),
    Abs(u8, u8),
    Ceiling(u8, u8),
    Cos(u8, u8),
    Truncate(u8, u8),
    Floor(u8, u8),
    Ln(u8, u8),
    Log(u8, u8),
    Neg(u8, u8),
    Round(u8, u8),
    Sin(u8, u8),
    Sqrt(u8, u8),
    Not(u8, u8),
    NotBool(u8, u8),
    Add(u8, u8, u8),
    Sub(u8, u8, u8),
    Mul(u8, u8, u8),
    Div(u8, u8, u8),
    Atan(u8, u8, u8),
    Exp(u8, u8, u8),
    Idiv(u8, u8, u8),
    Mod(u8, u8, u8),
    And(u8, u8, u8),
    Bitshift(u8, u8, u8),
    Eq(u8, u8, u8),
    Ne(u8, u8, u8),
    Ge(u8, u8, u8),
    Gt(u8, u8, u8),
    Le(u8, u8, u8),
    Lt(u8, u8, u8),
    Or(u8, u8, u8),
    Xor(u8, u8, u8),
    OrBool(u8, u8, u8),
    XorBool(u8, u8, u8),
    /// Skip this many instructions unless the register is true (not zero).
    SkipUnless(u8, u16),
    /// Skip this many instructions.
    Skip(u16),
    Fail,
}

impl Compiled {
    fn new(program: &[PostScriptOp], inputs: usize) -> Option<Self> {
        let mut compiler = Compiler {
            registers: Box::new([0.0; REGISTERS]),
            kinds: Vec::new(),
            constants: Vec::new(),
        };
        let mut stack = Vec::new();
        let mut code = Vec::new();

        if inputs > STACK_CAPACITY {
            code.push(Instruction::Fail);
        } else {
            for _ in 0..inputs {
                stack.push(compiler.register(Kind::Float, None)?);
            }
            code = compiler.block(program, &mut stack)?.0;
        }

        Some(Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            inputs,
            code,
            registers: compiler.registers,
            stack,
        })
    }

    fn run(&self, input: &[f32], output: &mut [f32]) -> Option<()> {
        REGISTER_FILE.with_borrow_mut(|(id, registers)| {
            if *id != self.id {
                *registers = *self.registers;
                *id = self.id;
            }
            self.execute(registers, input, output)
        })
    }

    fn execute(&self, r: &mut [f32; REGISTERS], input: &[f32], output: &mut [f32]) -> Option<()> {
        use Instruction::*;

        r[..input.len()].copy_from_slice(input);

        let mut pc = 0;
        while let Some(&instruction) = self.code.get(pc) {
            pc += 1;
            let i = usize::from;
            match instruction {
                Move(d, a) => r[i(d)] = r[i(a)],
                Abs(d, a) => r[i(d)] = r[i(a)].abs(),
                Ceiling(d, a) => r[i(d)] = r[i(a)].ceil(),
                Cos(d, a) => r[i(d)] = cos(r[i(a)]),
                Truncate(d, a) => r[i(d)] = r[i(a)].trunc(),
                Floor(d, a) => r[i(d)] = r[i(a)].floor(),
                Ln(d, a) => r[i(d)] = r[i(a)].ln(),
                Log(d, a) => r[i(d)] = r[i(a)].log10(),
                Neg(d, a) => r[i(d)] = -r[i(a)],
                Round(d, a) => r[i(d)] = r[i(a)].round(),
                Sin(d, a) => r[i(d)] = sin(r[i(a)]),
                Sqrt(d, a) => r[i(d)] = r[i(a)].sqrt(),
                Not(d, a) => r[i(d)] = not(r[i(a)]),
                NotBool(d, a) => r[i(d)] = bf(r[i(a)] == 0.0),
                Add(d, a, b) => r[i(d)] = r[i(a)] + r[i(b)],
                Sub(d, a, b) => r[i(d)] = r[i(a)] - r[i(b)],
                Mul(d, a, b) => r[i(d)] = r[i(a)] * r[i(b)],
                Div(d, a, b) => r[i(d)] = r[i(a)] / r[i(b)],
                Atan(d, a, b) => r[i(d)] = atan(r[i(a)], r[i(b)]),
                Exp(d, a, b) => r[i(d)] = r[i(a)].powf(r[i(b)]),
                Idiv(d, a, b) => r[i(d)] = idiv(r[i(a)], r[i(b)])?,
                Mod(d, a, b) => r[i(d)] = r[i(a)].rem(r[i(b)]),
                And(d, a, b) => r[i(d)] = and(r[i(a)], r[i(b)]),
                Bitshift(d, a, b) => r[i(d)] = bitshift(r[i(a)], r[i(b)]),
                Eq(d, a, b) => r[i(d)] = bf(r[i(a)] == r[i(b)]),
                Ne(d, a, b) => r[i(d)] = bf(r[i(a)] != r[i(b)]),
                Ge(d, a, b) => r[i(d)] = bf(r[i(a)] >= r[i(b)]),
                Gt(d, a, b) => r[i(d)] = bf(r[i(a)] > r[i(b)]),
                Le(d, a, b) => r[i(d)] = bf(r[i(a)] <= r[i(b)]),
                Lt(d, a, b) => r[i(d)] = bf(r[i(a)] < r[i(b)]),
                Or(d, a, b) => r[i(d)] = or(r[i(a)], r[i(b)]),
                Xor(d, a, b) => r[i(d)] = xor(r[i(a)], r[i(b)]),
                OrBool(d, a, b) => r[i(d)] = bf(r[i(a)] != 0.0 || r[i(b)] != 0.0),
                XorBool(d, a, b) => r[i(d)] = bf((r[i(a)] != 0.0) ^ (r[i(b)] != 0.0)),
                SkipUnless(c, n) => {
                    if r[i(c)] == 0.0 {
                        pc += usize::from(n);
                    }
                }
                Skip(n) => pc += usize::from(n),
                Fail => return None,
            }
        }

        let values = &self.stack[self.stack.len().checked_sub(output.len())?..];
        for (output, &register) in output.iter_mut().zip(values) {
            *output = r[usize::from(register)];
        }

        Some(())
    }
}

/// Whether a value is a boolean (held as 1 or 0) or a number, as the interpreter's
/// [`Argument`] would be. Comparisons give numbers; only `true`, `false` and `or`, `xor` and
/// `not` of booleans give booleans.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Kind {
    Float,
    Bool,
}

/// Whether a block of code reaches its end, or fails before it.
#[derive(PartialEq)]
enum Flow {
    Continues,
    Fails,
}

struct Compiler {
    registers: Box<[f32; REGISTERS]>,
    kinds: Vec<Kind>,
    /// The value of each register given by a number in the program (or `true` or `false`).
    constants: Vec<Option<f32>>,
}

impl Compiler {
    fn register(&mut self, kind: Kind, constant: Option<f32>) -> Option<u8> {
        let register = u8::try_from(self.kinds.len()).ok()?;
        self.kinds.push(kind);
        self.constants.push(constant);
        if let Some(value) = constant {
            self.registers[usize::from(register)] = value;
        }

        Some(register)
    }

    fn kind(&self, register: u8) -> Kind {
        self.kinds[usize::from(register)]
    }

    /// Compile `program`, given the registers on the stack as it starts, and leaving those
    /// on it at its end; `None` if it can't be compiled.
    fn block(
        &mut self,
        program: &[PostScriptOp],
        stack: &mut Vec<u8>,
    ) -> Option<(Vec<Instruction>, Flow)> {
        use Instruction as I;

        let mut code = Vec::new();

        macro_rules! fail {
            () => {{
                code.push(I::Fail);
                return Some((code, Flow::Fails));
            }};
        }
        macro_rules! pop {
            () => {
                match stack.pop() {
                    Some(register) => register,
                    None => fail!(),
                }
            };
        }
        macro_rules! push {
            ($register:expr) => {{
                let register = $register;
                if stack.len() == STACK_CAPACITY {
                    fail!();
                }
                stack.push(register);
            }};
        }
        macro_rules! one {
            ($instruction:path) => {{
                let a = pop!();
                let d = self.register(Kind::Float, None)?;
                code.push($instruction(d, a));
                push!(d);
            }};
        }
        macro_rules! two {
            ($instruction:path) => {{
                let b = pop!();
                let a = pop!();
                let d = self.register(Kind::Float, None)?;
                code.push($instruction(d, a, b));
                push!(d);
            }};
        }
        // A count `copy`, `index` or `roll` takes, which has to be a number in the program.
        macro_rules! count {
            () => {{
                let register = pop!();
                self.constants[usize::from(register)]?
            }};
        }

        for op in program {
            match op {
                PostScriptOp::Number(n) => push!(self.register(Kind::Float, Some(*n))?),
                PostScriptOp::True => push!(self.register(Kind::Bool, Some(1.0))?),
                PostScriptOp::False => push!(self.register(Kind::Bool, Some(0.0))?),
                PostScriptOp::Abs => one!(I::Abs),
                PostScriptOp::Ceiling => one!(I::Ceiling),
                PostScriptOp::Cos => one!(I::Cos),
                PostScriptOp::Cvi | PostScriptOp::Truncate => one!(I::Truncate),
                // A number, as it is (`Move` makes a boolean one).
                PostScriptOp::Cvr => one!(I::Move),
                PostScriptOp::Floor => one!(I::Floor),
                PostScriptOp::Ln => one!(I::Ln),
                PostScriptOp::Log => one!(I::Log),
                PostScriptOp::Neg => one!(I::Neg),
                PostScriptOp::Round => one!(I::Round),
                PostScriptOp::Sin => one!(I::Sin),
                PostScriptOp::Sqrt => one!(I::Sqrt),
                PostScriptOp::Add => two!(I::Add),
                PostScriptOp::Sub => two!(I::Sub),
                PostScriptOp::Mul => two!(I::Mul),
                PostScriptOp::Div => two!(I::Div),
                PostScriptOp::Atan => two!(I::Atan),
                PostScriptOp::Exp => two!(I::Exp),
                PostScriptOp::Idiv => two!(I::Idiv),
                PostScriptOp::Mod => two!(I::Mod),
                PostScriptOp::And => two!(I::And),
                PostScriptOp::Bitshift => two!(I::Bitshift),
                PostScriptOp::Eq => two!(I::Eq),
                PostScriptOp::Ne => two!(I::Ne),
                PostScriptOp::Ge => two!(I::Ge),
                PostScriptOp::Gt => two!(I::Gt),
                PostScriptOp::Le => two!(I::Le),
                PostScriptOp::Lt => two!(I::Lt),
                PostScriptOp::Not => {
                    let a = pop!();
                    let kind = self.kind(a);
                    let d = self.register(kind, None)?;
                    code.push(match kind {
                        Kind::Float => I::Not(d, a),
                        Kind::Bool => I::NotBool(d, a),
                    });
                    push!(d);
                }
                PostScriptOp::Or | PostScriptOp::Xor => {
                    let b = pop!();
                    let a = pop!();
                    let floats = self.kind(a) == Kind::Float && self.kind(b) == Kind::Float;
                    let kind = if floats { Kind::Float } else { Kind::Bool };
                    let d = self.register(kind, None)?;
                    code.push(match (op, floats) {
                        (PostScriptOp::Or, true) => I::Or(d, a, b),
                        (PostScriptOp::Or, false) => I::OrBool(d, a, b),
                        (_, true) => I::Xor(d, a, b),
                        (_, false) => I::XorBool(d, a, b),
                    });
                    push!(d);
                }
                PostScriptOp::If(then) => {
                    let condition = pop!();
                    let mut taken = stack.clone();
                    let (then, flow) = self.block(then, &mut taken)?;
                    if flow == Flow::Fails {
                        code.push(I::SkipUnless(condition, skip(&then)?));
                        code.extend(then);
                    } else {
                        let (moves, other_moves, merged) = self.merge(&taken, stack)?;
                        let over = then.len() + moves.len() + usize::from(!other_moves.is_empty());
                        code.push(I::SkipUnless(condition, u16::try_from(over).ok()?));
                        code.extend(then);
                        code.extend(moves);
                        if !other_moves.is_empty() {
                            code.push(I::Skip(skip(&other_moves)?));
                        }
                        code.extend(other_moves);
                        *stack = merged;
                    }
                }
                PostScriptOp::IfElse(then, otherwise) => {
                    let condition = pop!();
                    let mut first = stack.clone();
                    let (then, then_flow) = self.block(then, &mut first)?;
                    let mut second = stack.clone();
                    let (otherwise, otherwise_flow) = self.block(otherwise, &mut second)?;
                    let (moves, other_moves, merged) = match (&then_flow, &otherwise_flow) {
                        (Flow::Continues, Flow::Continues) => self.merge(&first, &second)?,
                        (Flow::Continues, Flow::Fails) => (Vec::new(), Vec::new(), first),
                        (Flow::Fails, _) => (Vec::new(), Vec::new(), second),
                    };
                    // The first branch skips the second unless it fails, which ends the run.
                    let skips = then_flow == Flow::Continues;
                    let over = then.len() + moves.len() + usize::from(skips);
                    code.push(I::SkipUnless(condition, u16::try_from(over).ok()?));
                    code.extend(then);
                    code.extend(moves);
                    if skips {
                        let over = otherwise.len() + other_moves.len();
                        code.push(I::Skip(u16::try_from(over).ok()?));
                    }
                    code.extend(otherwise);
                    code.extend(other_moves);
                    if then_flow == Flow::Fails && otherwise_flow == Flow::Fails {
                        return Some((code, Flow::Fails));
                    }
                    *stack = merged;
                }
                PostScriptOp::Copy => {
                    let n = count!() as u32 as usize;
                    let Some(start) = stack.len().checked_sub(n) else {
                        fail!();
                    };
                    for i in start..stack.len() {
                        push!(stack[i]);
                    }
                }
                PostScriptOp::Dup => {
                    let Some(&top) = stack.last() else {
                        fail!();
                    };
                    push!(top);
                }
                PostScriptOp::Exch => {
                    let b = pop!();
                    let a = pop!();
                    push!(b);
                    push!(a);
                }
                PostScriptOp::Index => {
                    let n = count!() as u32 as usize;
                    let Some(&value) = stack
                        .len()
                        .checked_sub(n + 1)
                        .and_then(|index| stack.get(index))
                    else {
                        fail!();
                    };
                    push!(value);
                }
                PostScriptOp::Pop => {
                    pop!();
                }
                PostScriptOp::Roll => {
                    let j = count!() as i32;
                    let n = count!() as u32 as usize;
                    let Some(start) = stack.len().checked_sub(n) else {
                        fail!();
                    };
                    roll(&mut stack[start..], j);
                }
            }
        }

        Some((code, Flow::Continues))
    }

    /// Where two ways through a branch leave the stack, the moves at the end of each that
    /// leave it the same, and the stack they leave; `None` if they leave it at different
    /// depths or with values of different kinds.
    fn merge(
        &mut self,
        a: &[u8],
        b: &[u8],
    ) -> Option<(Vec<Instruction>, Vec<Instruction>, Vec<u8>)> {
        if a.len() != b.len() {
            return None;
        }

        let (mut a_moves, mut b_moves, mut merged) = (Vec::new(), Vec::new(), Vec::new());
        for (&a, &b) in a.iter().zip(b) {
            if a == b {
                merged.push(a);
                continue;
            }
            if self.kind(a) != self.kind(b) {
                return None;
            }
            let register = self.register(self.kind(a), None)?;
            a_moves.push(Instruction::Move(register, a));
            b_moves.push(Instruction::Move(register, b));
            merged.push(register);
        }

        Some((a_moves, b_moves, merged))
    }
}

fn skip(code: &[Instruction]) -> Option<u16> {
    u16::try_from(code.len()).ok()
}

fn parse_procedure(data: &[u8]) -> Option<Vec<PostScriptOp>> {
    let mut r = Reader::new(data);
    parse_procedure_inner(&mut r)
}

fn parse_procedure_inner(r: &mut Reader<'_>) -> Option<Vec<PostScriptOp>> {
    let mut stack = ParseStack::new();

    let mut ops = vec![];
    r.skip_white_spaces_and_comments();
    r.forward_tag(b"{")?;

    loop {
        r.skip_white_spaces_and_comments();

        if r.peek_byte()? == b'}' {
            r.forward_tag(b"}")?;

            break;
        } else if r.peek_byte()? == b'{' {
            stack.push(parse_procedure_inner(r)?);
        } else {
            let op = PostScriptOp::from_reader(r, &mut stack)?;
            ops.push(op);
        }
    }

    Some(ops)
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum PostScriptOp {
    /// A number, as the `f32` it's pushed as.
    Number(f32),
    Abs,
    Add,
    Atan,
    Ceiling,
    Cos,
    Cvi,
    Cvr,
    Div,
    Exp,
    Floor,
    Idiv,
    Ln,
    Log,
    Mod,
    Mul,
    Neg,
    Round,
    Sin,
    Sqrt,
    Sub,
    Truncate,
    And,
    Bitshift,
    Eq,
    False,
    Ge,
    Gt,
    Le,
    Lt,
    Ne,
    Not,
    Or,
    True,
    Xor,
    If(Vec<Self>),
    IfElse(Vec<Self>, Vec<Self>),
    Copy,
    Dup,
    Exch,
    Index,
    Pop,
    Roll,
}

impl PostScriptOp {
    fn from_reader(r: &mut Reader<'_>, stack: &mut ParseStack) -> Option<Self> {
        let op = if let Some(n) = r.read::<Number>(&ReaderContext::dummy()) {
            // TODO: Support radix numbers
            Self::Number(n.as_f64() as f32)
        } else {
            let op = r.read::<content::Operator<'_>>(&ReaderContext::dummy())?;
            match op.as_ref() {
                b"abs" => Self::Abs,
                b"add" => Self::Add,
                b"atan" => Self::Atan,
                b"ceiling" => Self::Ceiling,
                b"cos" => Self::Cos,
                b"cvi" => Self::Cvi,
                b"cvr" => Self::Cvr,
                b"div" => Self::Div,
                b"exp" => Self::Exp,
                b"floor" => Self::Floor,
                b"idiv" => Self::Idiv,
                b"ln" => Self::Ln,
                b"log" => Self::Log,
                b"mod" => Self::Mod,
                b"mul" => Self::Mul,
                b"neg" => Self::Neg,
                b"round" => Self::Round,
                b"sin" => Self::Sin,
                b"sqrt" => Self::Sqrt,
                b"sub" => Self::Sub,
                b"truncate" => Self::Truncate,
                b"and" => Self::And,
                b"bitshift" => Self::Bitshift,
                b"eq" => Self::Eq,
                b"false" => Self::False,
                b"ge" => Self::Ge,
                b"gt" => Self::Gt,
                b"le" => Self::Le,
                b"lt" => Self::Lt,
                b"ne" => Self::Ne,
                b"not" => Self::Not,
                b"or" => Self::Or,
                b"true" => Self::True,
                b"xor" => Self::Xor,
                b"if" => Self::If(stack.pop()?),
                b"ifelse" => {
                    let s = stack.pop()?;
                    let f = stack.pop()?;
                    Self::IfElse(f, s)
                }
                b"copy" => Self::Copy,
                b"dup" => Self::Dup,
                b"exch" => Self::Exch,
                b"index" => Self::Index,
                b"pop" => Self::Pop,
                b"roll" => Self::Roll,
                _ => {
                    return None;
                }
            }
        };

        Some(op)
    }
}

#[cfg(test)]
mod tests {
    use crate::function::type4::{Compiled, PostScriptOp, Type4, parse_procedure};
    use crate::function::{Function, FunctionType, Interval, Repr};
    use std::f32::consts::LN_10;
    use std::sync::Arc;

    #[test]
    fn lex_1() {
        let program = b"{ copy dup 2.0 exch roll }";
        let parsed = parse_procedure(program).unwrap();

        assert_eq!(
            parsed,
            vec![
                PostScriptOp::Copy,
                PostScriptOp::Dup,
                PostScriptOp::Number(2.0),
                PostScriptOp::Exch,
                PostScriptOp::Roll,
            ]
        );
    }

    #[test]
    fn lex_3() {
        let program = b" {  {dup exch} if {0} {1} ifelse }";
        let parsed = parse_procedure(program).unwrap();

        assert_eq!(
            parsed,
            vec![
                PostScriptOp::If(vec![PostScriptOp::Dup, PostScriptOp::Exch]),
                PostScriptOp::IfElse(
                    vec![PostScriptOp::Number(0.0)],
                    vec![PostScriptOp::Number(1.0)]
                )
            ]
        );
    }

    fn op_impl(prog: &str, out: &[f32]) {
        let procedure = format!("{{{prog}}}");
        let procedure = parse_procedure(procedure.as_bytes()).unwrap();

        let type4 = Type4::from_program(procedure);
        let stack = type4.run(&[]).unwrap();
        let res = stack.items().iter().map(|i| i.as_f32()).collect::<Vec<_>>();

        assert_eq!(res.as_slice(), out);
    }

    #[test]
    fn op_abs() {
        op_impl("4.5 abs", &[4.5]);
        op_impl("-3 abs", &[3.0]);
        op_impl("0 abs", &[0.0]);
    }

    #[test]
    fn op_atan() {
        op_impl("0 1 atan", &[0.0]);
        op_impl("1 0 atan", &[90.0]);
        op_impl("-100 0 atan", &[270.0]);
        op_impl("4 4 atan", &[45.0]);
    }

    #[test]
    fn op_ceiling() {
        op_impl("3.2 ceiling", &[4.0]);
        op_impl("-4.8 ceiling", &[-4.0]);
        op_impl("99 ceiling", &[99.0]);
    }

    #[test]
    fn op_cos() {
        op_impl("0 cos", &[1.0]);
        // Should be zero, but floating point impreciseness.
        op_impl("90 cos", &[-4.371139e-8]);
    }

    #[test]
    fn op_cvi() {
        op_impl("-47.8 cvi", &[-47.0]);
        op_impl("520.9 cvi", &[520.0]);
    }

    #[test]
    fn op_cvr() {
        op_impl("-47.8 cvr", &[-47.8]);
        op_impl("520 cvr", &[520.0]);
    }

    #[test]
    fn op_div() {
        op_impl("3 2 div", &[1.5]);
        op_impl("4 2 div", &[2.0]);
    }

    #[test]
    fn op_exp() {
        op_impl("9 0.5 exp", &[3.0]);
        op_impl("-9 -1 exp", &[-0.11111111]);
    }

    #[test]
    fn op_floor() {
        op_impl("3.2 floor", &[3.0]);
        op_impl("-4.8 floor", &[-5.0]);
        op_impl("99 floor", &[99.0]);
    }

    #[test]
    fn op_idiv() {
        op_impl("3 2 idiv", &[1.0]);
        op_impl("4 2 idiv", &[2.0]);
        op_impl("-5 2 idiv", &[-2.0]);
    }

    #[test]
    fn op_ln() {
        op_impl("10 ln", &[LN_10]);
        op_impl("100 ln", &[4.6051702]);
    }

    #[test]
    fn op_log() {
        op_impl("10 log", &[1.0]);
        op_impl("100 log", &[2.0]);
    }

    #[test]
    fn op_mod() {
        op_impl("5 3 mod", &[2.0]);
        op_impl("5 2 mod", &[1.0]);
        op_impl("-5 3 mod", &[-2.0]);
    }

    #[test]
    fn op_mul() {
        op_impl("5 3 mul", &[15.0]);
        op_impl("-2 6 mul", &[-12.0]);
    }

    #[test]
    fn op_neg() {
        op_impl("4.5 neg", &[-4.5]);
        op_impl("-3 neg", &[3.0]);
    }

    #[test]
    fn op_round() {
        op_impl("3.2 round", &[3.0]);
        op_impl("6.5 round", &[7.0]);
        op_impl("-4.8 round", &[-5.0]);
        // TODO: This rounding doesn't match the PS spec.
        // op_impl("-6.5 round", &[-6.0]);
        op_impl("99 round", &[99.0]);
    }

    #[test]
    fn op_sin() {
        op_impl("0.0 sin", &[0.0]);
        op_impl("90.0 sin", &[1.0]);
    }

    #[test]
    fn op_sqrt() {
        op_impl("100 sqrt", &[10.0]);
    }

    #[test]
    fn op_sub() {
        op_impl("3 4 sub", &[-1.0]);
        op_impl("6 0 sub", &[6.0]);
    }

    #[test]
    fn op_truncate() {
        op_impl("3.2 truncate", &[3.0]);
        op_impl("-4.8 truncate", &[-4.0]);
        op_impl("99 truncate", &[99.0]);
    }

    #[test]
    fn op_and() {
        op_impl("true true and", &[1.0]);
        op_impl("true false and", &[0.0]);
        op_impl("false true and", &[0.0]);
        op_impl("false false and", &[0.0]);
        op_impl("99 1 and", &[1.0]);
        op_impl("52 7 and", &[4.0]);
    }

    #[test]
    fn op_bitshift() {
        op_impl("7 3 bitshift", &[56.0]);
        op_impl("142 -3 bitshift", &[17.0]);
    }

    #[test]
    fn op_eq() {
        op_impl("4.0 4 eq", &[1.0]);
        op_impl("-2.0 -3 eq", &[0.0]);
    }

    #[test]
    fn op_false() {
        op_impl("false", &[0.0]);
    }

    #[test]
    fn op_ge() {
        op_impl("4.2 4 ge", &[1.0]);
        op_impl("4.2 4.2 ge", &[1.0]);
        op_impl("4.2 6 ge", &[0.0]);
    }

    #[test]
    fn op_gt() {
        op_impl("4.2 4 gt", &[1.0]);
        op_impl("4.2 4.2 gt", &[0.0]);
        op_impl("4.2 6 gt", &[0.0]);
    }

    #[test]
    fn op_le() {
        op_impl("4.2 4 le", &[0.0]);
        op_impl("4.2 4.2 le", &[1.0]);
        op_impl("4.2 6 le", &[1.0]);
    }

    #[test]
    fn op_lt() {
        op_impl("4.2 4 lt", &[0.0]);
        op_impl("4.2 4.2 lt", &[0.0]);
        op_impl("4.2 6 lt", &[1.0]);
    }

    #[test]
    fn op_ne() {
        op_impl("3.0 3 ne", &[0.0]);
        op_impl("3.0 3.0 ne", &[0.0]);
        op_impl("3.0 3.1 ne", &[1.0]);
    }

    #[test]
    fn op_not() {
        op_impl("true not", &[0.0]);
        op_impl("false not", &[1.0]);
        op_impl("52 not", &[-53.0]);
    }

    #[test]
    fn op_or() {
        op_impl("true true or", &[1.0]);
        op_impl("true false or", &[1.0]);
        op_impl("false true or", &[1.0]);
        op_impl("false false or", &[0.0]);
        op_impl("17 5 or", &[21.0]);
    }

    #[test]
    fn op_xor() {
        op_impl("true true xor", &[0.0]);
        op_impl("true false xor", &[1.0]);
        op_impl("false true xor", &[1.0]);
        op_impl("false false xor", &[0.0]);
        op_impl("7 3 xor", &[4.0]);
        op_impl("12 3 xor", &[15.0]);
    }

    #[test]
    fn op_if() {
        op_impl("true { 1.0 } if", &[1.0]);
        op_impl("false { 1.0 } if", &[]);
    }

    #[test]
    fn op_ifelse() {
        op_impl("true { 1.0 } { 2.0 } ifelse", &[1.0]);
        op_impl("false { 1.0 } { 2.0 } ifelse", &[2.0]);
    }

    #[test]
    fn op_copy() {
        op_impl("1 2 3 2 copy", &[1.0, 2.0, 3.0, 2.0, 3.0]);
        op_impl("1 2 3 0 copy", &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn op_dup() {
        op_impl("1 2 3 dup", &[1.0, 2.0, 3.0, 3.0]);
    }

    #[test]
    fn op_exch() {
        op_impl("1 2 3 exch", &[1.0, 3.0, 2.0]);
    }

    #[test]
    fn op_index() {
        op_impl("1 2 3 4 0 index", &[1.0, 2.0, 3.0, 4.0, 4.0]);
        op_impl("1 2 3 4 3 index", &[1.0, 2.0, 3.0, 4.0, 1.0]);
    }

    #[test]
    fn op_pop() {
        op_impl("1 2 3 pop", &[1.0, 2.0]);
    }

    #[test]
    fn op_roll() {
        op_impl("1 2 3 3 -1 roll", &[2.0, 3.0, 1.0]);
        op_impl("1 2 3 3 1 roll", &[3.0, 1.0, 2.0]);
        op_impl("1 2 3 3 0 roll", &[1.0, 2.0, 3.0]);
        op_impl("1 2 3 3 5 roll", &[2.0, 3.0, 1.0]);
        op_impl("0 2 roll", &[]);
        op_impl(
            "1 2 3 4 5 6 7 5 2 roll",
            &[1.0, 2.0, 6.0, 7.0, 3.0, 4.0, 5.0],
        );
    }

    #[test]
    fn domain() {
        let procedure = parse_procedure(b"{  }").unwrap();

        let type4 = Function(Arc::new(Repr {
            domain: vec![
                Interval {
                    min: -5.0,
                    max: 5.0
                };
                3
            ],
            range: None,
            num_outputs: 3,
            function_type: FunctionType::Type4(Type4::from_program(procedure)),
        }));

        let res = type4.eval_one(&[-10.0, -2.0, 6.0]).unwrap();

        assert_eq!(res.as_slice(), &[-5.0, -2.0, 5.0]);
    }

    /// What a program gives interpreted and compiled (`None` for the second if it isn't),
    /// with NaNs alike whatever their bits.
    fn interpreted_and_compiled(
        program: &[PostScriptOp],
        input: &[f32],
        outputs: usize,
    ) -> (Option<Vec<u32>>, Option<Option<Vec<u32>>>) {
        let bits = |values: Vec<f32>| {
            values
                .iter()
                .map(|v| if v.is_nan() { u32::MAX } else { v.to_bits() })
                .collect()
        };
        let type4 = Type4::from_program(program.to_vec());
        let mut output = vec![0.0; outputs];
        let interpreted = type4
            .interpret(input, &mut output)
            .map(|()| bits(output.clone()));
        let compiled = Compiled::new(program, input.len()).map(|compiled| {
            let mut output = vec![0.0; outputs];
            compiled.run(input, &mut output).map(|()| bits(output))
        });

        (interpreted, compiled)
    }

    #[test]
    fn compiled_oklab_matches() {
        let program = parse_procedure(OKLAB).unwrap();
        assert!(Compiled::new(&program, 3).is_some());
        for l in 0..=20 {
            for a in 0..=20 {
                for b in 0..=20 {
                    let input = [l as f32 / 20.0, a as f32 / 20.0, b as f32 / 20.0];
                    let (interpreted, compiled) = interpreted_and_compiled(&program, &input, 3);
                    assert!(interpreted.is_some());
                    assert_eq!(compiled, Some(interpreted), "{input:?}");
                }
            }
        }
    }

    /// issue17065's conversion from `OKLab` to linear sRGB.
    const OKLAB: &[u8] = b"{ exch 0.4 sub exch 0.4 sub 2 index 2 index 2 index 0.2158037573 mul \
        exch 0.3963377774 mul add add dup dup mul mul 3 index 3 index 3 index -0.0638541728 mul \
        exch -0.1055613458 mul add add dup dup mul mul 4 index 4 index 4 index -1.2914855480 mul \
        exch -0.0894841775 mul add add dup dup mul mul 6 3 roll pop pop pop 2 index 2 index \
        2 index 0.2309699292 mul exch -3.3077115913 mul add exch 4.0767416621 mul add 3 index \
        3 index 3 index -0.3413193965 mul exch 2.6097574011 mul add exch -1.2684380046 mul add \
        4 index 4 index 4 index 1.7076147010 mul exch -0.7034186147 mul add exch \
        -0.0041960863 mul add 6 3 roll pop pop pop }";

    struct Rng(u64);

    impl Rng {
        fn below(&mut self, n: u64) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0 % n
        }

        fn pick<'a>(&mut self, items: &[&'a str]) -> &'a str {
            items[self.below(items.len() as u64) as usize]
        }
    }

    const NUMBERS: &[&str] = &[
        "0", "1", "-1", "2", "3", "0.5", "-2.5", "7", "100", "-0.0", "1000000", "0.001",
    ];
    const ONE_OPERAND: &[&str] = &[
        "abs", "ceiling", "cos", "cvi", "cvr", "floor", "ln", "log", "neg", "round", "sin", "sqrt",
        "truncate", "not",
    ];
    const TWO_OPERANDS: &[&str] = &[
        "add", "sub", "mul", "div", "atan", "exp", "idiv", "mod", "and", "bitshift", "eq", "ne",
        "ge", "gt", "le", "lt", "or", "xor",
    ];

    /// A random program, and how many more values it leaves on the stack than it finds
    /// (if it doesn't run out of them first).
    fn random_program(rng: &mut Rng, depth: u32, ops: u64) -> (String, i64) {
        let mut program = String::new();
        let mut effect = 0;
        for _ in 0..rng.below(ops) + 1 {
            let (op, op_effect) = match rng.below(24) {
                0..=4 => (rng.pick(NUMBERS).to_string(), 1),
                5 => (rng.pick(&["true", "false"]).to_string(), 1),
                6..=8 => (rng.pick(ONE_OPERAND).to_string(), 0),
                9..=12 => (rng.pick(TWO_OPERANDS).to_string(), -1),
                13 => {
                    let n = rng.below(4);
                    (format!("{n} copy"), n as i64)
                }
                14 => (format!("{} index", rng.below(4)), 1),
                15 => (
                    format!("{} {} roll", rng.below(5), rng.below(9) as i64 - 4),
                    0,
                ),
                16 => ("dup".to_string(), 1),
                17 => ("exch".to_string(), 0),
                18 => ("pop".to_string(), -1),
                // A count the program computes, which isn't compiled.
                19 if rng.below(8) == 0 => ("1 1 add index".to_string(), 1),
                20..=23 if depth < 3 => {
                    // Branches that mostly leave the stack as deep as each other, so that
                    // they're compiled; some don't.
                    let (then, then_effect) = random_program(rng, depth + 1, 6);
                    let (otherwise, otherwise_effect) = random_program(rng, depth + 1, 6);
                    let balance = |effect: i64, to: i64| {
                        let pad = if effect > to { " pop" } else { " 0" };
                        pad.repeat((effect - to).unsigned_abs() as usize)
                    };
                    if rng.below(2) == 0 {
                        let then = then + &balance(then_effect, 0);
                        (format!("{{{then}}} if"), -1)
                    } else {
                        let unbalanced = rng.below(6) == 0;
                        let otherwise = otherwise
                            + &balance(otherwise_effect, then_effect + i64::from(unbalanced));
                        (
                            format!("{{{then}}} {{{otherwise}}} ifelse"),
                            then_effect - 1,
                        )
                    }
                }
                _ => (rng.pick(NUMBERS).to_string(), 1),
            };
            program.push(' ');
            program.push_str(&op);
            effect += op_effect;
        }

        (program, effect)
    }

    /// Random programs give the same compiled as interpreted, failures included.
    #[test]
    fn compiled_programs_match() {
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let (mut compiled_count, mut succeeded, mut branched) = (0, 0, 0);
        let inputs = [0.0, 1.0, -1.0, 0.25, 3.7, -100.0, 1e10, 2.0];

        for _ in 0..20_000 {
            let n_inputs = rng.below(4) as usize;
            let (text, _) = random_program(&mut rng, 0, 16);
            let program = parse_procedure(format!("{{{text}}}").as_bytes()).unwrap();
            for _ in 0..4 {
                let input = (0..n_inputs)
                    .map(|_| inputs[rng.below(inputs.len() as u64) as usize])
                    .collect::<Vec<_>>();
                let outputs = rng.below(3) as usize + 1;
                let (interpreted, compiled) = interpreted_and_compiled(&program, &input, outputs);
                if let Some(compiled) = compiled {
                    compiled_count += 1;
                    succeeded += usize::from(compiled.is_some());
                    branched += usize::from(compiled.is_some() && text.contains(" if"));
                    assert_eq!(compiled, interpreted, "{program:?} on {input:?}");
                }
            }
        }

        // Most are compiled, many of those run to the end, and many of those branch.
        assert!(compiled_count > 50_000, "{compiled_count}");
        assert!(succeeded > 10_000, "{succeeded}");
        assert!(branched > 5_000, "{branched}");
    }

    #[test]
    fn compiled_failures() {
        let compile = |program: &str, inputs| {
            Compiled::new(&parse_procedure(program.as_bytes()).unwrap(), inputs)
        };
        // A branch that underflows fails only when it's taken.
        let program = compile("{ { pop pop pop } if 5 }", 2).unwrap();
        let mut output = [0.0];
        assert!(program.run(&[1.0, 0.0], &mut output).is_some());
        assert_eq!(output, [5.0]);
        assert!(program.run(&[1.0, 1.0], &mut output).is_none());
        // Dividing by zero fails rather than panicking.
        assert!(
            compile("{ 1 0 idiv }", 0)
                .unwrap()
                .run(&[], &mut output)
                .is_none()
        );
        // Computed counts and branches of different depths are interpreted.
        assert!(compile("{ 1 1 add index }", 3).is_none());
        assert!(compile("{ { 1 } if }", 1).is_none());
        // Too many values for the stack fail as the interpreter does.
        let deep = format!("{{ {} }}", "1 ".repeat(101));
        assert!(compile(&deep, 0).unwrap().run(&[], &mut output).is_none());
    }
}
